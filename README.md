# AC Quantum Sensing Array Simulation

This repository contains a lightweight Python simulation for visualizing collective signal accumulation in an array of quantum sensors under AC field sensing.

The code models an array of sensors with local signal amplitudes \(B_i\) and phases \(\phi_i\). For a collective GHZ-like sensing picture, the local AC signals are compressed into an effective complex phasor sum,

$$
B_{\rm eff}=\left|\sum_i B_i e^{i\phi_i}\right|.
$$

This effective coupling controls the distinguishability between the no-signal and signal-evolved sensor states through the trace distance,

$$
D(T)=\left|\sin\frac{\Theta(T)}{2}\right|,
\qquad
\Theta(T)\sim B_{\rm eff}T.
$$

The simulation compares three representative cases:

1. Homogeneous signal: same $\(B_i=B\)$, same $\(\phi_i=\phi\)$
2. Amplitude disorder: random $\(B_i\)$, same $\(\phi_i=\phi\)$
3. Phase disorder: same $\(B_i=B\)$, random $\(\phi_i\)$

The goal is to illustrate how spatial phase disorder suppresses coherent collective enhancement. In the ideal homogeneous case, the effective signal scales as $\(B_{\rm eff}\sim NB\)$, while for random phases it typically scales like a random walk, $\(B_{\rm eff}\sim \sqrt{N}B\)$. This directly increases the sensing time required to reach a fixed trace-distance detection threshold.

The repository also includes heuristic comparisons of conventional frequency scanning and quantum-search-sensing-inspired scaling:

$$
\tau_{\rm conv}\sim \frac{\Delta\omega}{B_{\rm eff}^2},
\qquad
\tau_{\rm QSS}\sim \frac{\sqrt{\Delta\omega}}{B_{\rm eff}^{3/2}}.
$$

These scalings are intended as a toy-level baseline, not a full implementation of the quantum search sensing oracle.
