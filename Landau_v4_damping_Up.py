#!/usr/bin/env python3

"""Landau V4: fazno mesanje istorijskih promena i rang cele kombinacije."""

import argparse
import csv
from pathlib import Path
import numpy as np

DEFAULTS = (
    "/data/loto7_4698_k80.csv",
    "/data/loto7_4698_k80_loto_2971.csv",
    "/data/loto7_4698_k80_loto_plus_1727.csv",
)

N, K, WINDOW, MEMORY, OSCILLATORS = 39, 7, 64, 32, 8
START = MEMORY+1
VMAX = (.5, 1., 2.)
ALPHAS = (.01, .1, 1.)

j = np.arange(8)
DCT = np.vstack((
    np.ones(8)/np.sqrt(8),
    np.sqrt(2/8)*np.cos(
        np.pi*np.arange(1, 8)[:, None]
        * (j[None, :]+.5)/8
    )
))
MODES = DCT[1:]
WAVENUMBERS = np.pi*np.arange(1, 8)/8


def load(path):
    rows = []

    with Path(path).open(
        newline="", encoding="utf-8-sig"
    ) as stream:
        for line, row in enumerate(csv.reader(stream), 1):
            if not row or all(not v.strip() for v in row):
                continue

            try:
                values = [int(v.strip()) for v in row]
            except ValueError as error:
                raise ValueError(
                    f"Red {line}: ocekujem cele brojeve."
                ) from error

            if (
                len(values) != K
                or values != sorted(values)
                or len(set(values)) != K
                or not all(1 <= v <= N for v in values)
            ):
                raise ValueError(
                    f"Red {line}: neispravna sortirana "
                    "7/39 kombinacija."
                )

            rows.append(values)

    return np.asarray(rows, dtype=int).reshape(-1, K)


def gaps(rows):
    rows = np.atleast_2d(rows)

    edges = np.column_stack((
        np.zeros(len(rows)),
        rows,
        np.full(len(rows), 40)
    ))

    return np.diff(edges, axis=1)-1.


def excitations(rows, reference):
    return (
        (gaps(rows)-gaps(reference)) @ MODES.T / 5.
    )


def replacements(row):
    unused = np.setdiff1d(np.arange(1, 40), row)
    result = np.tile(row, (K*len(unused), 1))

    for i in range(K):
        result[i*len(unused):(i+1)*len(unused), i] = unused

    return np.sort(result, axis=1)


def frequencies(changes, vmax):
    # Brzine poticu iz empirijskih strukturnih promena.
    # Nema nametnute uniformne ili normalne raspodele.
    speeds = np.sort(np.linalg.norm(changes, axis=1))

    indices = np.rint(
        np.linspace(0, len(speeds)-1, OSCILLATORS)
    ).astype(int)

    return (
        WAVENUMBERS[:, None]
        * (vmax*speeds[indices])[None, :]
    )


def transport(amplitude, omega, time):
    return amplitude*np.exp(1j*omega*time)


def latent_energy(amplitude):
    return float(np.sum(np.abs(amplitude)**2))


def phase_state(changes, t, omega):
    # changes[r] je promena od reda r do reda r+1.
    # Za ciljni red t poslednja dostupna promena
    # jeste changes[t-2].
    end = t-1
    begin = max(0, end-MEMORY)

    impulses = changes[begin:end]
    ages = t-(np.arange(begin, end)+1)

    particles = (
        impulses[:, :, None]/np.sqrt(OSCILLATORS)
    )
    evolved = particles*np.exp(
        1j*ages[:, None, None]*omega[None, :, :]
    )

    return evolved.sum(axis=0)/np.sqrt(MEMORY)


def context(state):
    signal = state.sum(axis=1)/np.sqrt(OSCILLATORS)
    return np.r_[signal.real, signal.imag]


def design(rows, reference, c):
    e = excitations(rows, reference)

    interaction = (
        e[:, :, None]*c[None, None, :]
    ).reshape(len(e), 98)

    # Direktan skalarni skor cele kombinacije.
    # Fazni signal se ne pretvara u prosecne polozaje.
    return np.column_stack((e, e*e, interaction))


def prepare(history, vmax):
    if len(history) < WINDOW+START:
        raise ValueError(
            f"Potrebno je najmanje "
            f"{WINDOW+START} prethodnih redova."
        )

    history = history[-(WINDOW+START):]

    changes = (
        np.diff(gaps(history), axis=0) @ MODES.T / 5.
    )
    omega = frequencies(changes, vmax)

    gram = np.zeros((112, 112))
    target = np.zeros(112)

    for t in range(START, len(history)):
        c = context(phase_state(changes, t, omega))

        actual = design(
            history[t:t+1], history[t-1], c
        )
        alternatives = design(
            replacements(history[t]), history[t-1], c
        )

        contrasts = actual-alternatives

        gram += (
            contrasts.T @ contrasts/len(contrasts)
        )
        target += contrasts.sum(axis=0)/len(contrasts)

    gram /= WINDOW
    target /= WINDOW

    candidates = np.unique(
        np.vstack(
            [history[START:]]
            + [
                replacements(row)
                for row in history[START:]
            ]
        ),
        axis=0
    )

    state = phase_state(changes, len(history), omega)
    features = design(
        candidates, history[-1], context(state)
    )

    return (
        gram, target, candidates, features,
        state, omega
    )


def predict(prepared, alpha):
    gram, target, candidates, features = prepared[:4]

    coef = np.linalg.solve(
        gram+alpha*np.eye(112), target
    )
    scores = features @ coef

    if not np.all(np.isfinite(scores)):
        raise ArithmeticError(
            "Neispravan skor faznog modela."
        )

    best = int(np.argmax(scores))

    return (
        tuple(map(int, candidates[best])),
        float(scores[best])
    )


def hits(prediction, actual):
    return len(
        set(map(int, prediction))
        & set(map(int, actual))
    )


def summarize(label, scores):
    print(
        f"{label}: prosek={np.mean(scores):.4f}; "
        f"pogodaka 0..7="
        f"{np.bincount(scores, minlength=8).tolist()}",
        flush=True
    )


def run_file(path, validation_count=48, test_count=64):
    rows = load(path)
    split = len(rows)-test_count
    begin = split-validation_count

    if (
        validation_count < 16
        or test_count < 16
        or begin < WINDOW+START
    ):
        raise ValueError(
            "Premalo istorije ili manje od "
            "16 validacionih/test kola."
        )

    settings = tuple(
        (v, a) for v in VMAX for a in ALPHAS
    )
    validation = [[] for _ in settings]

    print(
        f"\nUlaz: {Path(path).resolve()}; "
        f"redova={len(rows)}",
        flush=True
    )

    for t in range(begin, split):
        index = 0

        for vmax in VMAX:
            prepared = prepare(rows[:t], vmax)

            for alpha in ALPHAS:
                combo, _ = predict(prepared, alpha)
                validation[index].append(
                    hits(combo, rows[t])
                )
                index += 1

        if (t-begin+1) % 16 == 0:
            print(
                f"Validacija: "
                f"{t-begin+1}/{validation_count}",
                flush=True
            )

    means = [np.mean(s) for s in validation]
    chosen = int(np.argmax(means))
    vmax, alpha = settings[chosen]

    print(
        f"Validacija: redovi {begin+1}..{split}; "
        f"vmax={vmax:g}; alpha={alpha:g}; "
        f"prosek={means[chosen]:.4f}",
        flush=True
    )

    scores, previous = [], []

    # Vmax i alpha ostaju fiksni tokom testa.
    # Svako kolo koristi iskljucivo prethodne redove.
    for t in range(split, len(rows)):
        combo, _ = predict(
            prepare(rows[:t], vmax), alpha
        )

        scores.append(hits(combo, rows[t]))
        previous.append(hits(rows[t-1], rows[t]))

        if (t-split+1) % 16 == 0:
            print(
                f"Test: {t-split+1}/{test_count}",
                flush=True
            )

    print(f"Test: redovi {split+1}..{len(rows)}")
    summarize("LANDAU V4 DAMPING", scores)
    summarize("Prethodna kombinacija", previous)

    prepared = prepare(rows, vmax)
    combo, score = predict(prepared, alpha)

    state, omega = prepared[4:]
    evolved = transport(state, omega, 1.)

    print(
        f"Poslednji red: "
        f"{','.join(map(str, rows[-1]))}"
    )
    print(
        f"SLEDECI RED {len(rows)+1}: "
        f"{','.join(map(str, combo))}"
    )
    print(
        f"Skor={score:.10f}; "
        f"kandidata={len(prepared[2])}"
    )
    print(
        "Slobodan fazni transport bez nove pobude: "
        f"energija t=0 {latent_energy(state):.10f}; "
        f"t=1 {latent_energy(evolved):.10f}"
    )
    print(
        "Izbor: poslednje 64 kombinacije "
        "i sve njihove zamene jednog broja."
    )
    print(
        "Fazno mesanje je strukturna analogija; "
        "skor nije verovatnoca."
    )


def self_test():
    assert np.allclose(DCT @ DCT.T, np.eye(8))
    assert np.allclose(MODES.sum(axis=1), 0.)

    # Kontrolni primer faznog ponistavanja
    # uz ocuvanu energiju pojedinacnih komponenti.
    omega = (
        2*np.pi*np.arange(OSCILLATORS)[None, :]
        / OSCILLATORS
    )
    amplitude = np.ones(
        (1, OSCILLATORS), dtype=complex
    )
    evolved = transport(amplitude, omega, 1.)

    assert abs(evolved.sum()) < 1e-10
    assert np.isclose(
        latent_energy(amplitude),
        latent_energy(evolved)
    )

    base = np.array([
        [1,2,3,4,5,6,7],
        [4,12,18,28,35,36,37],
        [33,34,35,36,37,38,39],
        [2,7,12,18,24,30,39]
    ])
    history = np.tile(base, (30, 1))

    changes = (
        np.diff(gaps(history), axis=0) @ MODES.T / 5.
    )
    omega = frequencies(changes, 1.)

    changed = changes.copy()
    changed[69:] = 0.

    assert np.array_equal(
        phase_state(changes, 70, omega),
        phase_state(changed, 70, omega)
    )

    e = excitations(base, base[0])
    assert np.allclose(
        gaps(base[0])+5*e @ MODES,
        gaps(base)
    )

    for vmax in VMAX:
        p = prepare(history, vmax)

        assert np.linalg.eigvalsh(p[0]).min() >= -1e-8

        for time in (-97., 1., 1000.):
            assert np.isclose(
                latent_energy(p[4]),
                latent_energy(
                    transport(p[4], p[5], time)
                ),
                rtol=1e-12
            )

        for alpha in ALPHAS:
            result = predict(p, alpha)

            assert result == predict(
                prepare(history, vmax), alpha
            )
            assert any(
                np.array_equal(result[0], x)
                for x in p[2]
            )

    altered = history.copy()
    altered[-1] = base[0]

    assert (
        predict(prepare(history[:-1], 1.), .1)
        == predict(prepare(altered[:-1], 1.), .1)
    )

    print(
        "Self-test OK: fazno mesanje, energija, "
        "rekonstrukcija i hronologija."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="*")
    parser.add_argument("--validation", type=int, default=48)
    parser.add_argument("--test", type=int, default=64)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    for path in args.csv or DEFAULTS:
        run_file(path, args.validation, args.test)


if __name__ == "__main__":
    main()



"""
Ulaz: /data/loto7_4698_k80.csv; redova=4698
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: redovi 4587..4634; vmax=1; alpha=1; prosek=1.5625
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 4635..4698
LANDAU V4 DAMPING: prosek=1.4062; pogodaka 0..7=[13, 24, 16, 10, 1, 0, 0, 0]
Prethodna kombinacija: prosek=1.2188; pogodaka 0..7=[13, 30, 16, 4, 1, 0, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 4699: 8,x,10,y,13,z,35
Skor=0.9251476223; kandidata=14396
Slobodan fazni transport bez nove pobude: energija t=0 13.4868291557; t=1 13.4868291557
Izbor: poslednje 64 kombinacije i sve njihove zamene jednog broja.
Fazno mesanje je strukturna analogija; skor nije verovatnoca.



Ulaz: /data/loto7_4698_k80_loto_2971.csv; redova=2971
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: redovi 2860..2907; vmax=1; alpha=0.1; prosek=1.3750
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 2908..2971
LANDAU V4 DAMPING: prosek=1.2812; pogodaka 0..7=[14, 25, 19, 5, 1, 0, 0, 0]
Prethodna kombinacija: prosek=1.2969; pogodaka 0..7=[11, 30, 16, 7, 0, 0, 0, 0]
Poslednji red: 18,20,21,24,29,34,38
SLEDECI RED 2972: 1,x,19,y,24,z,39
Skor=0.4664211223; kandidata=14392
Slobodan fazni transport bez nove pobude: energija t=0 9.4257382069; t=1 9.4257382069
Izbor: poslednje 64 kombinacije i sve njihove zamene jednog broja.
Fazno mesanje je strukturna analogija; skor nije verovatnoca.



Ulaz: /data/loto7_4698_k80_loto_plus_1727.csv; redova=1727
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: redovi 1616..1663; vmax=0.5; alpha=0.01; prosek=1.4583
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 1664..1727
LANDAU V4 DAMPING: prosek=1.0156; pogodaka 0..7=[21, 23, 18, 2, 0, 0, 0, 0]
Prethodna kombinacija: prosek=1.2969; pogodaka 0..7=[15, 23, 18, 8, 0, 0, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 1728: 1,x,14,y,17,z,36
Skor=0.5978874041; kandidata=14388
Slobodan fazni transport bez nove pobude: energija t=0 8.8741131400; t=1 8.8741131400
Izbor: poslednje 64 kombinacije i sve njihove zamene jednog broja.
Fazno mesanje je strukturna analogija; skor nije verovatnoca.
"""



"""
Fazno mešanje sada služi za rangiranje celih kombinacija, bez predikcije prosečnih položaja. 
Brzine komponenti izvode se iz stvarnih strukturnih promena.

Fajl	                Sledeća kombinacija	Prosek pogodaka na 64 kola	Prethodna kombinacija
Objedinjeni, red 4699	8,x,10,y,13,z,35	1,4062	                    1,2188
Loto, red 2972	        1,x,19,y,24,z,39	1,2812	                    1,2969
Loto Plus, red 1728	    1,x,14,y,17,z,36	1,0156	                    1,2969

Prednost na ovom testu postoji samo za objedinjeni fajl. 
Provereni su fazno poništavanje, očuvanje energije pri slobodnom transportu, ponovljivost i hronologija.
"""
