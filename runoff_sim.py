"""
Monte Carlo simulation of the 2026 Brazil presidential runoff (Lula vs Flávio Bolsonaro).

Model
-----
1. Lula and Bolsonaro first-round voters vote for the same candidate again (100% retention).
2. Each third-party camp i (N_i first-round voters) splits into
       A = abstain / blank / null,  B = Bolsonaro,  L = Lula
   with theta_i ~ Dirichlet(lambda * alpha_i)  (priors from the "Vote Migration Priors" doc),
   and counts n_i | theta_i ~ Multinomial(N_i, theta_i).
3. Valid votes rise by N between rounds. Because some third-party voters drop out, the
   cohort of genuinely new runoff voters is  N_new = N + sum_i n_{A,i}.
   Lula's share of them is a fixed constant theta_new (no prior): we sweep it to see what
   Lula needs. Counts ~ Binomial(N_new, theta_new).
4. Margin = (Lula - Bolsonaro) / total valid runoff votes.

Output: P(Lula margin >= target), P(Lula wins), and the margin distribution, at each theta_new.
"""

import argparse
import numpy as np

# ---------------------------------------------------------------- first round constants
LULA_R1 = 53_879_538
BOLSO_R1 = 56_104_503

THIRD_PARTY_VOTES = {
    "Cury":    3_448_569,
    "Santos":  2_675_887,
    "Caiado":  2_605_148,
    "Zema":      326_488,
    "Martins":   122_911,
    "Dias":       43_103,
}

N_INCREASE = 4876126            # increase in valid votes, round 2 vs round 1


DEFAULT_THETA = 0.80              # Lula's share of new runoff voters
TARGET_MARGIN = 0.0358            # Lula +3.58 pp

# ---------------------------------------------------------------- Dirichlet priors (A, B, L)
PRIORS_BASE = {
    "Cury":    (8.0, 16.0, 17.4),
    "Santos":  (14.1, 11.6, 8.4),
    "Caiado":  (11.9, 20.8, 8.9),
    "Zema":    (6.8, 17.6, 5.6),
    "Martins": (2.4, 0.9, 5.5),
    "Dias":    (2.0, 0.6, 3.6),
}


POLLED_CAMPS = ("Cury", "Santos", "Caiado")   # priors fitted through the poll likelihood


def adjust_priors(priors, lula_bias=0.0, abstain_shift=0.0, camps=POLLED_CAMPS):
    """Shift selected camps' prior means in log-odds space, keeping each kappa fixed.

    lula_bias     > 0: polls overstated Lula vs Flavio (the doc's b_L; +0.25 = 2022-calibrated
                       variant), so the L/B odds are multiplied by exp(-lula_bias).
    abstain_shift > 0: more abstention / blank / null than polls show (opposite sign of the
                       doc's b_A), so the A odds are multiplied by exp(abstain_shift).
    """
    out = {}
    for camp, alpha in priors.items():
        a = np.asarray(alpha, dtype=float)
        if camp in camps and (lula_bias or abstain_shift):
            kappa, m = a.sum(), a / a.sum()
            m_a = 1 / (1 + (1 - m[0]) / m[0] * np.exp(-abstain_shift))
            lb_odds = m[2] / m[1] * np.exp(-lula_bias)
            m_b = (1 - m_a) / (1 + lb_odds)
            a = kappa * np.array([m_a, m_b, 1 - m_a - m_b])
        out[camp] = tuple(a)
    return out


def simulate_camps(n_sims=2_000_000, priors=PRIORS_BASE, lam=1.0, seed=2026):
    """Round-1 loyalists plus the six third-party camps. Returns (lula, bolso, abstained):
    vote counts before any new voters join, and the third-party voters who sit out."""
    rng = np.random.default_rng(seed)

    lula = np.full(n_sims, LULA_R1, dtype=np.int64)
    bolso = np.full(n_sims, BOLSO_R1, dtype=np.int64)
    abstained = np.zeros(n_sims, dtype=np.int64)

    # third-party camps: Dirichlet -> Multinomial (via sequential binomials)
    for camp, n_i in THIRD_PARTY_VOTES.items():
        alpha = lam * np.asarray(priors[camp], dtype=float)
        theta = rng.dirichlet(alpha, size=n_sims)               # columns: A, B, L
        n_a = rng.binomial(n_i, theta[:, 0])
        p_b_given_valid = theta[:, 1] / (theta[:, 1] + theta[:, 2])
        n_b = rng.binomial(n_i - n_a, p_b_given_valid)
        n_l = n_i - n_a - n_b
        abstained += n_a
        bolso += n_b
        lula += n_l

    return lula, bolso, abstained


def add_new_voters(camps, theta_new=DEFAULT_THETA, n_increase=N_INCREASE, seed=2026):
    """Give Lula a fixed share theta_new of the new-voter cohort. The cohort draws share one
    seed, so sweeping theta_new over the same camps changes nothing else."""
    lula, bolso, abstained = camps
    rng = np.random.default_rng([seed, 1])

    # new runoff voters = N + those who abstained
    n_new = n_increase + abstained
    new_l = rng.binomial(n_new, theta_new)
    lula = lula + new_l
    bolso = bolso + n_new - new_l

    total_valid = lula + bolso
    expected_total = LULA_R1 + BOLSO_R1 + sum(THIRD_PARTY_VOTES.values()) + n_increase
    assert np.all(total_valid == expected_total), "vote accounting mismatch"

    margin = (lula - bolso) / total_valid
    return {"margin": margin, "abstained": abstained, "n_new": n_new,
            "total_valid": expected_total}


def simulate(n_sims=2_000_000, priors=PRIORS_BASE, lam=1.0, n_increase=N_INCREASE,
             theta_new=DEFAULT_THETA, seed=2026):
    camps = simulate_camps(n_sims, priors, lam, seed)
    return add_new_voters(camps, theta_new, n_increase, seed)


def required_theta(camps, target=0.0, n_increase=N_INCREASE):
    """Per simulation, the theta_new at which Lula's expected margin equals target (a fraction of
    valid votes); it can fall outside [0, 1] when the camps alone put the target out of reach.
    P(margin >= target | theta_new) is the share of simulations with required_theta <= theta_new,
    up to the binomial noise of the cohort, which is negligible."""
    lula, bolso, abstained = camps
    n_new = n_increase + abstained
    total_valid = LULA_R1 + BOLSO_R1 + sum(THIRD_PARTY_VOTES.values()) + n_increase
    return (1 + (target * total_valid - (lula - bolso)) / n_new) / 2


def summarize(res, target=TARGET_MARGIN, label=""):
    m = res["margin"]
    n = m.size
    p_target = np.mean(m >= target)
    p_win = np.mean(m > 0)
    se = np.sqrt(p_target * (1 - p_target) / n)
    q = np.quantile(m, [0.05, 0.25, 0.5, 0.75, 0.95])

    print(f"\n=== {label} ===")
    print(f"Simulations                    : {n:,}")
    print(f"Total valid runoff votes       : {res['total_valid']:,}")
    print(f"Mean third-party abstainers    : {res['abstained'].mean():,.0f}")
    print(f"Mean new-voter cohort          : {res['n_new'].mean():,.0f}")
    print(f"P(Lula margin >= {target:+.2%})    : {p_target:.4f}  (MC s.e. {se:.4f})")
    print(f"P(Lula wins)                   : {p_win:.4f}")
    print(f"Lula margin mean               : {m.mean():+.2%}")
    print("Lula margin quantiles 5/25/50/75/95: "
          + "  ".join(f"{x:+.2%}" for x in q))
    return p_target


def sweep(thetas, lam, sims, target, seed):
    """One row per theta_new, all on the same third-party draws."""
    camps = simulate_camps(sims, PRIORS_BASE, lam, seed)
    print(f"\n=== base, λ={lam}: sweep over Lula's share of new voters ===")
    print(f"{'theta_new':>9}  {'P(win)':>7}  {f'P(>={target:+.2%})':>12}  {'mean margin':>11}")
    for t in thetas:
        m = add_new_voters(camps, t, seed=seed)["margin"]
        print(f"{t:9.3f}  {np.mean(m > 0):7.4f}  {np.mean(m >= target):12.4f}  {m.mean():+11.2%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=2_000_000)
    ap.add_argument("--lam", type=float, default=1.0, help="Dirichlet flexibility knob λ")
    ap.add_argument("--theta", type=float, nargs="+", default=[DEFAULT_THETA],
                    help="Lula's share of new runoff voters; several values print a sweep table")
    ap.add_argument("--target", type=float, default=TARGET_MARGIN)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--all", action="store_true", help="run the λ=1.0 and λ=0.5 variants")
    args = ap.parse_args()

    for lam in (1.0, 0.5) if args.all else (args.lam,):
        if len(args.theta) > 1:
            sweep(args.theta, lam, args.sims, args.target, args.seed)
        else:
            theta = args.theta[0]
            res = simulate(args.sims, PRIORS_BASE, lam, theta_new=theta, seed=args.seed)
            summarize(res, args.target, f"base, λ={lam}, theta_new={theta:.3f}")


if __name__ == "__main__":
    main()
