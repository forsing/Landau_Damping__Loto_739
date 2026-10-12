#!/usr/bin/env python3

"""LANDAU V4 — DAMPING: conservative phase mixing of structural waves.""" 

import argparse
import csv
import itertools
from pathlib import Path
import numpy as np



DEFAULT = "/Users/4c/Desktop/GHQ/data/loto7_4698_k80.csv"
# DEFAULT = "/Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_2971.csv"
# DEFAULT = "/Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_plus_1727.csv"



N, K, WINDOW, OSCILLATORS = 39, 7, 256, 24
SIZE = K + 1

j = np.arange(SIZE)
DCT = np.vstack((
    np.ones(SIZE)/np.sqrt(SIZE),
    np.sqrt(2/SIZE)*np.cos(
        np.pi*np.arange(1, SIZE)[:, None]
        * (j[None, :]+.5)/SIZE
    ),
))
MODES = DCT[1:]
WAVENUMBERS = np.pi*np.arange(1, SIZE)/SIZE


def load(path):
    rows = []
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        for line, row in enumerate(csv.reader(f), 1):
            if not row or all(not v.strip() for v in row):
                continue
            try:
                numbers = [int(v.strip()) for v in row]
            except ValueError as error:
                raise ValueError(
                    f"Red {line}: ocekujem sedam celih brojeva."
                ) from error

            if (len(numbers) != K or numbers != sorted(numbers)
                    or len(set(numbers)) != K
                    or not all(1 <= v <= N for v in numbers)):
                raise ValueError(
                    f"Red {line}: neispravna sortirana 7/39 kombinacija."
                )
            rows.append(numbers)

    return np.asarray(rows, dtype=float).reshape(-1, K)


def gaps(x, n=N):
    return np.diff(np.r_[0., x, float(n+1)]) - 1.


def observe(model, times):
    offset, amplitude, omega = model
    times = np.atleast_1d(times)
    transport = np.exp(
        1j*times[:, None, None]*omega[None, :, :]
    )
    return (
        offset[None, :]
        + (amplitude[None, :, :]*transport).sum(axis=2).real
    )


def latent_energy(model, time):
    _, amplitude, omega = model
    evolved = amplitude*np.exp(1j*omega*time)
    return float(np.sum(np.abs(evolved)**2))


def target_from_modes(amplitudes):
    target_gaps = (
        (N-K)/SIZE + np.asarray(amplitudes) @ MODES
    )
    return np.cumsum(target_gaps[:-1]+1), target_gaps


def fit(history, vmax, alpha):
    # Samo poznati prethodni redovi.
    history = history[-WINDOW:]
    gap_history = np.asarray([
        gaps(row) for row in history
    ])
    y = (gap_history-(N-K)/SIZE) @ MODES.T

    velocity = (
        vmax*np.arange(1, OSCILLATORS+1)/OSCILLATORS
    )
    omega = WAVENUMBERS[:, None]*velocity[None, :]
    times = np.arange(-len(history), 0, dtype=float)
    phase = times[:, None, None]*omega[None, :, :]

    design = np.concatenate((
        np.ones((len(history), K, 1)),
        np.cos(phase),
        np.sin(phase),
    ), axis=2)

    a = design.transpose(1, 0, 2)
    penalty = np.diag(np.r_[
        0., np.ones(2*OSCILLATORS)
    ])

    gram = a.transpose(0, 2, 1) @ a
    rhs = a.transpose(0, 2, 1) @ y.T[:, :, None]
    coefficients = np.linalg.solve(
        gram+alpha*penalty, rhs
    )[:, :, 0]

    amplitude = (
        coefficients[:, 1:OSCILLATORS+1]
        - 1j*coefficients[:, OSCILLATORS+1:]
    )
    model = (coefficients[:, 0], amplitude, omega)

    predicted_modes = observe(model, 0)[0]
    target, target_gaps = target_from_modes(
        predicted_modes
    )

    fitted = np.einsum(
        "lki,ki->lk", design, coefficients
    )
    residual_gaps = (y-fitted) @ MODES
    residual_positions = np.cumsum(
        residual_gaps[:, :-1], axis=1
    )

    ps = np.maximum(
        1., np.median(np.abs(residual_positions), axis=0)
    )
    gs = np.maximum(
        1., np.median(np.abs(residual_gaps), axis=0)
    )
    return target, ps, gs, model, target_gaps


def cost(combo, target, ps, gs, n=N):
    return float(
        (np.abs(np.asarray(combo)-target)/ps).sum()
        + (
            np.abs(gaps(combo, n)-gaps(target, n))/gs
        ).sum()
    )


def decode(target, ps, gs, n=N):
    # Tacan globalni minimum nad svim rastucim kombinacijama.
    k = len(target)
    target_gaps = gaps(target, n)
    table = np.full((k, n+1), np.inf)
    parent = np.full((k, n+1), -1, dtype=int)

    for v in range(1, n-k+2):
        table[0, v] = (
            abs(v-target[0])/ps[0]
            + abs(v-1-target_gaps[0])/gs[0]
        )

    for j in range(1, k):
        for v in range(j+1, n-k+j+2):
            previous = np.arange(j, v)
            choices = (
                table[j-1, previous]
                + np.abs(
                    v-previous-1-target_gaps[j]
                )/gs[j]
            )
            best = int(np.argmin(choices))
            parent[j, v] = previous[best]
            table[j, v] = (
                choices[best]+abs(v-target[j])/ps[j]
            )

    ends = np.arange(k, n+1)
    scores = (
        table[k-1, ends]
        + np.abs(n-ends-target_gaps[k])/gs[k]
    )
    best = int(np.argmin(scores))
    v = int(ends[best])
    combo = [v]

    for j in range(k-1, 0, -1):
        v = int(parent[j, v])
        combo.append(v)

    return tuple(reversed(combo)), float(scores[best])


def hits(prediction, actual):
    return len(
        set(map(int, prediction)) & set(map(int, actual))
    )


def summarize(label, scores):
    print(
        f"{label}: prosek={np.mean(scores):.4f}; "
        f"pogodaka 0..7="
        f"{np.bincount(scores, minlength=8).tolist()}"
    )


def self_test():
    assert np.allclose(DCT @ DCT.T, np.eye(SIZE))
    assert np.allclose(MODES.sum(axis=1), 0)

    rows = np.tile(np.array([
        [1, 2, 3, 4, 5, 6, 7],
        [4, 12, 18, 28, 35, 36, 37],
        [33, 34, 35, 36, 37, 38, 39],
        [2, 7, 12, 18, 24, 30, 39],
    ], dtype=float), (50, 1))

    target, ps, gs, model, predicted_gaps = fit(
        rows, 1., 10.
    )
    energy = latent_energy(model, 0)

    for time in (-256., 1., 17., 1000.):
        assert np.isclose(
            latent_energy(model, time),
            energy,
            rtol=1e-12,
            atol=1e-12,
        )

    assert np.isclose(predicted_gaps.sum(), N-K)
    assert np.allclose(gaps(target), predicted_gaps)
    assert decode(target, ps, gs) == decode(target, ps, gs)

    # Signal se ponistava, dok energija komponenti ostaje ocuvana.
    omega = (
        WAVENUMBERS[:, None]
        * np.arange(1, OSCILLATORS+1)[None, :]
        / OSCILLATORS
    )
    amplitudes = np.zeros_like(omega, dtype=complex)
    amplitudes[0] = 1.
    mixing_model = (np.zeros(K), amplitudes, omega)
    mixing_time = 2*np.pi/WAVENUMBERS[0]

    assert np.isclose(
        observe(mixing_model, 0)[0, 0], OSCILLATORS
    )
    assert abs(
        observe(mixing_model, mixing_time)[0, 0]
    ) < 1e-10
    assert np.isclose(
        latent_energy(mixing_model, 0),
        latent_energy(mixing_model, mixing_time),
    )

    target = np.array([8., 2., 5.])
    ps = np.array([1., 1.4, 1.1])
    gs = np.array([1., 1.3, 1.2, 1.])

    combo, score = decode(target, ps, gs, n=9)
    exact = min(
        cost(c, target, ps, gs, n=9)
        for c in itertools.combinations(range(1, 10), 3)
    )
    assert np.isclose(
        score, exact, rtol=1e-12, atol=1e-12
    )
    assert np.isclose(
        score, cost(combo, target, ps, gs, n=9)
    )

    print(
        "Self-test OK: fazno mesanje, "
        "ocuvanje energije i globalni minimum."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", default=DEFAULT)
    parser.add_argument("--validation", type=int, default=48)
    parser.add_argument("--test", type=int, default=64)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    rows = load(args.csv)
    if args.validation < 16 or args.test < 16:
        parser.error(
            "Validation i test moraju imati bar 16 kola."
        )

    split = len(rows)-args.test
    begin = split-args.validation
    if begin < 160:
        parser.error(
            "Premalo istorije za zadatu validaciju i test."
        )

    configurations = tuple(itertools.product(
        (.5, 1., 2.), (1., 10., 100.)
    ))
    validation = [[] for _ in configurations]

    print(
        f"Ulaz: {Path(args.csv).resolve()}; "
        f"redova={len(rows)}",
        flush=True,
    )

    for t in range(begin, split):
        for index, (vmax, alpha) in enumerate(configurations):
            target, ps, gs, _, _ = fit(
                rows[:t], vmax, alpha
            )
            combo, _ = decode(target, ps, gs)
            validation[index].append(hits(combo, rows[t]))

    means = [
        np.mean(values) for values in validation
    ]
    best = int(np.argmax(means))
    vmax, alpha = configurations[best]

    print(
        f"Validacija: redovi {begin+1}..{split}; "
        f"vmax={vmax:g}; alpha={alpha:g}; "
        f"prosek={means[best]:.4f}",
        flush=True,
    )

    model_scores = []
    previous_scores = []
    background_scores = []

    for t in range(split, len(rows)):
        target, ps, gs, _, _ = fit(
            rows[:t], vmax, alpha
        )
        combo, _ = decode(target, ps, gs)

        model_scores.append(hits(combo, rows[t]))
        previous_scores.append(hits(rows[t-1], rows[t]))

        background = np.mean([
            gaps(row)
            for row in rows[max(0, t-WINDOW):t]
        ], axis=0)
        reference, _ = decode(
            np.cumsum(background[:-1]+1),
            np.ones(K),
            np.ones(K+1),
        )
        background_scores.append(hits(reference, rows[t]))

    print(f"Test: redovi {split+1}..{len(rows)}")
    summarize("LANDAU V4 DAMPING", model_scores)
    summarize("Prethodna kombinacija", previous_scores)
    summarize("Pozadina bez talasa", background_scores)

    target, ps, gs, model, predicted_gaps = fit(
        rows, vmax, alpha
    )
    combo, score = decode(target, ps, gs)

    print(
        f"Poslednji red: "
        f"{','.join(map(str, rows[-1].astype(int)))}"
    )
    print(
        f"SLEDECI RED {len(rows)+1}: "
        f"{','.join(map(str, combo))}"
    )
    print(
        f"Trosak projekcije={score:.10f}; "
        "globalni minimum potvrden."
    )

    e0 = latent_energy(model, 0)
    e1 = latent_energy(model, 1)
    print(
        f"Energija komponenti: "
        f"t=0 {e0:.10f}; t=1 {e1:.10f}"
    )
    print(
        "Model koristi fazno mesanje bez trenja; "
        "trosak rangiranja nije verovatnoca."
    )


if __name__ == "__main__":
    main()



"""
Ulaz: /Users/4c/Desktop/GHQ/data/loto7_4698_k80.csv; redova=4698
Validacija: redovi 4587..4634; vmax=0.5; alpha=10; prosek=1.4792
Test: redovi 4635..4698
LANDAU V4 DAMPING: prosek=1.1875; pogodaka 0..7=[15, 28, 16, 4, 1, 0, 0, 0]
Prethodna kombinacija: prosek=1.2188; pogodaka 0..7=[13, 30, 16, 4, 1, 0, 0, 0]
Pozadina bez talasa: prosek=1.5312; pogodaka 0..7=[11, 22, 19, 10, 2, 0, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 4699: 7,12,17,23,30,31,37
Trosak projekcije=2.2835022868; globalni minimum potvrden.
Energija komponenti: t=0 39.0188618985; t=1 39.0188618985
Model koristi fazno mesanje bez trenja; trosak rangiranja nije verovatnoca.



Ulaz: /Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_2971.csv; redova=2971
Validacija: redovi 2860..2907; vmax=2; alpha=10; prosek=1.4792
Test: redovi 2908..2971
LANDAU V4 DAMPING: prosek=1.3594; pogodaka 0..7=[14, 23, 19, 6, 2, 0, 0, 0]
Prethodna kombinacija: prosek=1.2969; pogodaka 0..7=[11, 30, 16, 7, 0, 0, 0, 0]
Pozadina bez talasa: prosek=1.2812; pogodaka 0..7=[12, 29, 17, 5, 1, 0, 0, 0]
Poslednji red: 18,20,21,24,29,34,38
SLEDECI RED 2972: 7,15,19,27,31,32,33
Trosak projekcije=2.0050834256; globalni minimum potvrden.
Energija komponenti: t=0 35.5687051418; t=1 35.5687051418
Model koristi fazno mesanje bez trenja; trosak rangiranja nije verovatnoca.



Ulaz: /Users/4c/Desktop/GHQ/data/loto7_4698_k80_loto_plus_1727.csv; redova=1727
Validacija: redovi 1616..1663; vmax=2; alpha=100; prosek=1.4167
Test: redovi 1664..1727
LANDAU V4 DAMPING: prosek=1.1875; pogodaka 0..7=[15, 29, 14, 5, 1, 0, 0, 0]
Prethodna kombinacija: prosek=1.2969; pogodaka 0..7=[15, 23, 18, 8, 0, 0, 0, 0]
Pozadina bez talasa: prosek=1.2344; pogodaka 0..7=[13, 27, 20, 4, 0, 0, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 1728: 6,10,16,21,27,33,37
Trosak projekcije=1.1489766844; globalni minimum potvrden.
Energija komponenti: t=0 13.8776012524; t=1 13.8776012524
Model koristi fazno mesanje bez trenja; trosak rangiranja nije verovatnoca.
"""



"""
Landau 1946 On vibrations electron plasma original pdf phase mixing collisionless damping

LANDAU DAMPING - how waves in a plasma lose energy without collisions

LANDAU DAMPING, zasnovan na faznom mešanju: doprinosi pojedinačnih talasa ostaju očuvani, dok se njihov zajednički signal menja zbog razilaženja faza. 

V4 će učiti amplitude i faze talasnih komponenti. Njihovo dalje kretanje biće bez trenja; sledeći raspored dobiće se iz njihovog zajedničkog signala. Provera će obuhvatiti i očuvanje energije pojedinačnih komponenti.
V4 je prošao proveru faznog poništavanja signala uz očuvanu energiju komponenti. Na poslednja 64 kola ostvario je 1,1875 pogodaka po kolu, naspram 1,5312 za pozadinu bez talasa. 

Koristi fazno mešanje: uči amplitude i faze komponenti, čiji se zajednički signal menja dok njihova pojedinačna energija ostaje očuvana. Inspiracija je Landauov rad o elektronskoj plazmi.

Predlog za red 4699:
7, 12, 17, 23, 30, 31, 37

Na poslednja 64 kola:
Metod	Prosečan broj pogodaka
V4	1,1875
Prethodna kombinacija	1,2188
Pozadina bez talasa	1,5312

V4 na ovom testu nije pokazao prednost. Provereni su fazno poništavanje, očuvanje energije, hronologija, ponovljivost i globalni minimum troška izbora kombinacije.
"""
