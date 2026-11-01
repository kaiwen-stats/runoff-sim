"""
Interactive front-end for runoff_sim.py.

Run:  .venv/bin/streamlit run app.py
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import runoff_sim as rs

st.set_page_config(page_title="Runoff Simulator", page_icon="🗳️", layout="wide",
                   initial_sidebar_state="collapsed")

# neo-brutalist palette (mirrored as CSS variables in style.css)
INK, PAPER, RED, YELLOW, BLUE, SAND = "#111111", "#F2EEE3", "#FF4B33", "#FFD60A", "#3D5AFE", "#CFC9B6"
FONT = "Space Mono, monospace"

# ---------------------------------------------------------------- defaults (= the base case)
# theta: Lula's share of new voters, in %, held fixed in each run (no prior on it).
BIAS_SHIFT = 0.25      # size of the poll-bias adjustment (log-odds); +0.25 is the doc's 2022 variant
ABSTAIN_SHIFT = 0.35   # size of the abstention adjustment (log-odds), as in the report's "more abstention" scenario
# Labels are short because they sit side by side in a narrow panel; the full wording lives in the tooltips.
BIAS_OPTIONS = {"Bolsonaro": -BIAS_SHIFT, "Neither": 0.0, "Lula": BIAS_SHIFT}                 # "Polls overstate ..."
ABSTAIN_OPTIONS = {"Lower": -ABSTAIN_SHIFT, "As polled": 0.0, "Higher": ABSTAIN_SHIFT}        # "Abstention vs polls"

DEFAULTS = dict(theta=80.0, lam=1.0, bias_choice="Neither", abstain_choice="As polled",
                target=3.58, sims=2_000_000, seed=2026, extend_all=False)

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


# ---------------------------------------------------------------- cached simulation
def _priors(lula_bias, abstain_shift, extend_all):
    camps = tuple(rs.PRIORS_BASE) if extend_all else rs.POLLED_CAMPS
    return rs.adjust_priors(rs.PRIORS_BASE, lula_bias, abstain_shift, camps)


@st.cache_data(show_spinner=False, max_entries=16)
def run_camps(n_sims, seed, lam, lula_bias, abstain_shift, extend_all):
    """The expensive part, independent of theta: moving the theta slider never re-runs it."""
    pri = _priors(lula_bias, abstain_shift, extend_all)
    return tuple(c.astype(np.int32) for c in rs.simulate_camps(n_sims, pri, lam, seed))


def run(camps, theta, seed):
    res = rs.add_new_voters(camps, theta, seed=seed)
    return dict(margin=res["margin"].astype(np.float32),
                abstained=float(res["abstained"].mean()), n_new=float(res["n_new"].mean()))


def theta_curve(camps, target, thetas):
    """P(Lula wins) and P(margin >= target) at every theta, from each simulation's required theta."""
    return [np.searchsorted(np.sort(rs.required_theta(camps, t)), thetas, side="right") / camps[0].size
            for t in (0.0, target)]


# ---------------------------------------------------------------- html helpers
def html(s):
    st.html(s)


def section(num, title, color):
    html(f'<div class="sec"><span class="num {color}">{num}</span>{title}</div>')


def card_head(title, note="", color="ink"):
    html(f'<div class="card-head {color}"><span>{title}</span><em>{note}</em></div>')


def segment(cls, label, p):
    """One zone of a proportional bar; the label drops out when the zone is too narrow to hold it."""
    if p < 0.001:
        return ""
    text = f"{label} <b>{p:.1%}</b>" if p >= 0.14 else f"<b>{p:.0%}</b>" if p >= 0.045 else ""
    return f'<div class="seg {cls}" style="flex:{p:.6f}" title="{label}: {p:.2%}">{text}</div>'


html(f"<style>{(Path(__file__).parent / 'style.css').read_text()}</style>")

# ---------------------------------------------------------------- masthead
html("""
<div class="masthead">
  <div class="title-block">
    <h1>Lula <mark>Victory</mark> Simulation<span class="stamp">2026</span></h1>
  </div>
  <div class="howto"><b>HOW IT WORKS</b>Move a dial. The whole runoff is replayed hundreds of thousands
  of times and the odds redraw instantly.</div>
</div>
""")

console_col, results_col = st.columns([1, 2.4], gap="large")

# ---------------------------------------------------------------- control panel
with console_col:
    with st.container(key="console"):
        html('<div class="console-head"><span>CONTROL PANEL</span><span class="live"><i></i>LIVE</span></div>')

        section("01", "NEW VOTERS", "red")
        st.slider("Lula share of new voters, θ", 40.0, 100.0, step=1.0, format="%d%%", key="theta",
                  help="The share of new runoff voters who pick Lula over Bolsonaro, held fixed in every "
                       "simulated runoff. \"New voters\" are people who didn't cast a valid vote in round 1, "
                       "including those who take the place of third-party voters staying home. Instead of "
                       "guessing it, move the dial and watch what Lula's odds become. Base case: 80%.")
        cohort = st.empty()   # filled after the simulation, once the cohort size is known

        section("02", "MIGRATION PRIORS", "yellow")
        st.slider("Flexibility λ", 0.1, 3.0, step=0.05, key="lam",
                  help="How much we trust the poll-based estimates of where third-party voters go. Below 1: "
                       "less trust, so results are more spread out. Above 1: more trust. The average split "
                       "stays the same. 1 is the base case.")
        st.radio("Polls overstate…", list(BIAS_OPTIONS), key="bias_choice", horizontal=True,
                 help="Whether polls got third-party voters wrong. \"Neither\" takes them at face "
                      "value. If polls overstate Lula, fewer of these voters go to him and more to Bolsonaro "
                      "(his odds against Bolsonaro drop by about a fifth, which matches the 2022 experience). "
                      "The other option is the reverse. Applies to Cury, Santos and Caiado voters unless you "
                      "tick the box under Engine room.")
        st.radio("Abstention vs polls", list(ABSTAIN_OPTIONS), key="abstain_choice", horizontal=True,
                 help="Whether the number of third-party voters who stay home or vote blank matches what polls "
                      "suggest (\"As polled\"). Lower or Higher shifts the odds of staying home by about two-fifths "
                      "(for example, 20% staying home becomes about 26%). Since people who stay home are "
                      "replaced by new voters, this also changes the size of the new-voter group. Applies to the "
                      "same voters as above.")

        section("03", "THE BET", "blue")
        st.number_input("Target margin (Lula lead, pp)", step=0.05, format="%.2f", key="target",
                        help="The size of Lula's lead, in percentage points of valid votes, that you want to "
                             "test. The result shows the chance he wins by at least this much.")

        with st.expander("ENGINE ROOM"):
            st.select_slider("Simulations", [100_000, 200_000, 500_000, 1_000_000, 2_000_000], key="sims")
            st.number_input("Seed", step=1, key="seed")
            st.checkbox("Also adjust Zema, Martins and Dias voters", key="extend_all",
                        help="By default the poll-bias and abstention adjustments apply only to Cury, Santos and "
                             "Caiado voters, whose behavior was estimated from polls. Tick this to apply them "
                             "to Zema, Martins and Dias voters as well.")
        st.button("↺ RESET TO BASE CASE", on_click=lambda: st.session_state.update(DEFAULTS),
                  use_container_width=True)

S = st.session_state
lula_bias, abstain_shift = BIAS_OPTIONS[S.bias_choice], ABSTAIN_OPTIONS[S.abstain_choice]
theta = S.theta / 100
camp_args = (int(S.sims), int(S.seed), S.lam, lula_bias, abstain_shift, S.extend_all)
base_camp_args = (int(S.sims), int(S.seed), DEFAULTS["lam"], 0.0, 0.0, S.extend_all)
is_base = camp_args == base_camp_args and S.theta == DEFAULTS["theta"]

with st.spinner("Simulating…"):
    camps = run_camps(*camp_args)
    cur = run(camps, theta, int(S.seed))
    base = cur if is_base else run(run_camps(*base_camp_args), DEFAULTS["theta"] / 100, int(S.seed))

cohort.html(f'<div class="readout"><span>New-voter cohort · average per runoff</span>'
            f'<b>{cur["n_new"] / 1e6:.1f} million</b>'
            f'<small>Lula nets {(2 * theta - 1) * cur["n_new"] / 1e6:+.2f} million votes from them</small></div>')

target = S.target / 100
m = cur["margin"]
p_t, p_w = float(np.mean(m >= target)), float(np.mean(m > 0))
p_t0 = float(np.mean(base["margin"] >= target))
se = float(np.sqrt(p_t * (1 - p_t) / m.size))
p_b = 1 - p_w                                    # Bolsonaro wins
p_red = p_t if target > 0 else p_w               # Lula wins, at or past the target
p_yellow = max(p_w - p_red, 0.0)                 # Lula wins, short of the target

# ---------------------------------------------------------------- results
with results_col:
    # headline numbers
    delta = ('<span class="tag">BASE CASE</span>' if is_base else
             f'<span class="tag ink">{"▲" if p_t >= p_t0 else "▼"} {abs(p_t - p_t0) * 100:.1f} PP VS BASE CASE</span>')
    html(f"""
    <div class="stats">
      <div class="stat yellow">
        <div class="stat-label">P(Lula wins | θ = {S.theta:.0f}%)</div>
        <div class="stat-value">{p_w:.1%}</div>
        <div class="stat-foot"><span class="tag">AT ANY MARGIN</span></div>
      </div>
      <div class="stat red">
        <div class="stat-label">P(margin ≥ {S.target:+.2f} pp | θ = {S.theta:.0f}%)</div>
        <div class="stat-value">{p_t:.1%}</div>
        <div class="stat-foot">{delta}<span title="Random-simulation error">±{se:.2%} MC ERROR</span></div>
      </div>
    </div>
    """)

    # margin distribution
    with st.container(key="chartcard"):
        card_head("MARGIN DISTRIBUTION", "each bar = share of simulated runoffs in that margin bin", "red")
        lo, hi = np.quantile(m, [0.001, 0.999])
        lo, hi = min(lo, 0.0) - 0.002, max(hi, target) + 0.002
        counts, edges = np.histogram(m, bins=90, range=(lo, hi))
        mid, width = (edges[:-1] + edges[1:]) / 2, edges[1] - edges[0]
        dens = counts / counts.sum()

        # One trace per outcome, so the legend is the readout: each colored zone carries its own probability.
        zones = [("BOLSONARO WINS", BLUE, mid < 0, p_b)]
        if target > 0:
            zones += [("LULA, SHORT OF TARGET", YELLOW, (mid >= 0) & (mid < target), p_yellow),
                      (f"LULA ≥ {S.target:+.2f} PP", RED, mid >= target, p_red)]
        else:
            zones += [("LULA WINS", RED, mid >= 0, p_w)]

        fig = go.Figure()
        for name, color, in_zone, p in zones:
            fig.add_trace(go.Bar(
                x=mid[in_zone] * 100, y=dens[in_zone] * 100, width=width * 100, name=f"{name}  <b>{p:.1%}</b>",
                marker_color=color, marker_line=dict(color=INK, width=1.5),
                hovertemplate="margin %{x:.2f} pp<br>%{y:.2f}% of simulations<extra></extra>"))
        fig.add_vline(x=S.target, line=dict(color=INK, width=4))
        fig.add_vline(x=0, line=dict(color=INK, width=2, dash="dot"))
        near_right = target > mid[0] + 0.7 * (mid[-1] - mid[0])   # label flips to the left of the line near the edge
        fig.add_annotation(x=S.target, xanchor="right" if near_right else "left", yref="paper", y=1,
                           yanchor="bottom", showarrow=False, bgcolor="#FFFFFF", borderwidth=3, bordercolor=INK,
                           borderpad=5, font=dict(family=FONT, size=13, color=INK),
                           text=f"<b>TARGET {S.target:+.2f} PP</b>")
        fig.update_layout(
            height=454, margin=dict(l=16, r=16, t=84, b=16), bargap=0, barmode="overlay", showlegend=True,
            legend=dict(orientation="h", x=0, xanchor="left", yref="container", y=1, yanchor="top",
                        bgcolor="#FFFFFF", bordercolor=INK, borderwidth=3, itemclick=False, itemdoubleclick=False,
                        font=dict(family=FONT, size=12, color=INK)),
            font=dict(family=FONT, color=INK, size=12), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFDF6",
            hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=INK, font=dict(family=FONT, color=INK)),
            xaxis=dict(title="LULA MARGIN, PP OF VALID VOTES", ticksuffix=" pp", showgrid=False, zeroline=False,
                       showline=True, linecolor=INK, linewidth=3, ticks="outside", tickwidth=3, ticklen=6,
                       automargin=True),
            yaxis=dict(title="% OF SIMS PER BIN", ticksuffix="%", gridcolor="rgba(17,17,17,0.14)",
                       griddash="dot", zeroline=False, showline=True, linecolor=INK, linewidth=3,
                       ticks="outside", tickwidth=3, ticklen=6, automargin=True))
        st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})

    # what Lula needs: the odds at every theta
    with st.container(key="thetacard"):
        card_head("WHAT LULA NEEDS", "his odds if new voters split this way, everything else as set at left", "yellow")
        grid = np.linspace(0.4, 1.0, 121)
        win_curve, target_curve = theta_curve(camps, target, grid)

        def half_note(curve, what):
            i = np.argmax(curve >= 0.5)   # first grid theta at or past a coin flip; both curves rise with theta
            return (f"{what} in half of runoffs once θ reaches <b>{grid[i]:.1%}</b>." if curve[i] >= 0.5 else
                    f"{what} in only <b>{curve[-1]:.0%}</b> of runoffs even at θ = 100%.")

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=grid * 100, y=win_curve * 100, mode="lines", line=dict(color=BLUE, width=4),
            name="LULA WINS", hovertemplate="θ %{x:.1f}%<br>Lula wins in %{y:.1f}% of runs<extra></extra>"))
        fig2.add_trace(go.Scatter(
            x=grid * 100, y=target_curve * 100, mode="lines", line=dict(color=RED, width=4),
            name=f"MARGIN ≥ {S.target:+.2f} PP",
            hovertemplate="θ %{x:.1f}%<br>target reached in %{y:.1f}% of runs<extra></extra>"))
        fig2.add_trace(go.Scatter(
            x=[S.theta, S.theta], y=[p_w * 100, p_t * 100], mode="markers",
            marker=dict(size=14, color=[BLUE, RED], line=dict(color=INK, width=3)),
            hoverinfo="skip", showlegend=False))
        fig2.add_hline(y=50, line=dict(color=INK, width=1.5, dash="dot"))
        fig2.add_vline(x=S.theta, line=dict(color=INK, width=4))
        fig2.add_annotation(x=S.theta, xanchor="left" if S.theta < 70 else "right", yref="paper", y=1,
                            yanchor="bottom", showarrow=False, bgcolor="#FFFFFF", borderwidth=3, bordercolor=INK,
                            borderpad=5, font=dict(family=FONT, size=13, color=INK),
                            text=f"<b>θ = {S.theta:.0f}%</b>")
        fig2.update_layout(
            height=380, margin=dict(l=16, r=16, t=62, b=16), showlegend=True,
            legend=dict(x=0.015, xanchor="left", y=0.97, yanchor="top", bgcolor="#FFFFFF", bordercolor=INK,
                        borderwidth=3, font=dict(family=FONT, size=12, color=INK)),
            font=dict(family=FONT, color=INK, size=12), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFDF6",
            hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=INK, font=dict(family=FONT, color=INK)),
            xaxis=dict(title="LULA'S SHARE OF NEW VOTERS, θ", ticksuffix="%", range=[40, 100], showgrid=False,
                       zeroline=False, showline=True, linecolor=INK, linewidth=3, ticks="outside", tickwidth=3,
                       ticklen=6, automargin=True),
            yaxis=dict(title="% OF SIMULATED RUNOFFS", ticksuffix="%", range=[0, 102], gridcolor="rgba(17,17,17,0.14)",
                       griddash="dot", zeroline=False, showline=True, linecolor=INK, linewidth=3,
                       ticks="outside", tickwidth=3, ticklen=6, automargin=True))
        st.plotly_chart(fig2, width="stretch", theme=None, config={"displayModeBar": False})
        html(f'<p class="note">{half_note(win_curve, "Lula wins")} '
             f'{half_note(target_curve, f"A margin ≥ {S.target:+.2f} pp arrives")}</p>')

    # priors in effect
    with st.container(key="priorscard"):
        card_head("PRIORS IN EFFECT", "where each third-party camp's voters go, on average", "yellow")
        pri = _priors(lula_bias, abstain_shift, S.extend_all)
        rows = ""
        for camp, n_i in rs.THIRD_PARTY_VOTES.items():
            a = S.lam * np.asarray(pri[camp])
            mean = a / a.sum()
            net = n_i * (mean[2] - mean[1])
            bar = "".join(segment(c, "", p) for c, p in zip(("sand", "blue", "red"), mean))
            rows += (f'<div class="row"><div class="camp">{camp}</div><div class="votes">{n_i:,}</div>'
                     f'<div class="splitbar mini">{bar}</div>'
                     f'<div class="net {"red" if net >= 0 else "blue"}">{net:+,.0f}</div></div>')
        html(f"""
        <div class="priors">
          <div class="row hdr"><div>Camp</div><div>R1 votes</div>
            <div class="key"><i class="sand"></i>Abstain<i class="blue"></i>Bolsonaro<i class="red"></i>Lula</div>
            <div>Net to Lula</div></div>
          {rows}
          <p class="note">Expected net = N × (Lula mean − Bolsonaro mean), before new voters.</p>
        </div>
        """)

    # assumptions
    with st.container(key="printcard"):
        card_head("THE FINE PRINT", "model assumptions", "blue")
        html("""
        <div class="fine">
          <div><b>1</b>Lula and Bolsonaro first-round voters <em>repeat their vote</em> (100% retention).</div>
          <div><b>2</b>Each third-party camp splits into <em>abstain / Bolsonaro / Lula</em> with
            θ ~ Dirichlet(λ·α), then counts ~ Multinomial.</div>
          <div><b>3</b>The valid-vote increase <em>N</em> plus third-party abstainers forms the new-voter cohort;
            Lula takes a fixed share <em>θ</em> of it, set by you, counts ~ Binomial. The runoff is not
            given a prior for θ: the curve shows what each value would mean.</div>
          <div><b>4</b>Margin = <em>(Lula − Bolsonaro) / total valid votes</em>.</div>
        </div>
        """)
