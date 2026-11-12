# Exact fixed-point certificates for reversible TT phase oracles

## Objects and scope

Let $V$ be a real table on a finite product grid. Let $A_j(x_j)$ be an open-boundary TT of rectangular real matrices, with first and last bond dimension one. Write $A(x)=A_1(x_1)\cdots A_d(x_d)$ and assume $\|V-A\|_\infty\le\tau$. The code interprets each supplied finite binary64 number as the exact dyadic rational it represents. For a materialized reference table, `exact_tt_values` evaluates $A$ with integer dot products and exact dyadic denominators, so the maximum residual is exact for those inputs. This contract excludes error in computing the input table from a continuous learned model. A theorem-backed sup-norm approximation bound can be supplied in place of enumeration.

Quantize $A_j$ entrywise to $B_j=2^{-f_j}Q_j$ by nearest-even rounding. Let $s_0=0$, select $0\le s_j\le s_{j-1}+f_j=:g_j$, and put $k_j=g_j-s_j$. For integer row vectors, define

$$z_0=(1),\qquad y_j=z_{j-1}Q_j(x_j),\qquad z_j=\lfloor 2^{-k_j}y_j\rfloor$$

entrywise. The implemented scalar is $\widehat V(x)=2^{-s_d}z_d$. The floor is signed and rounds negative values toward $-\infty$. No product term is individually rounded. Setting $k_j=0$ removes that stage's rounding entirely.

## Signed intervals and clean computation

Suppose the preceding retained integer component $a$ lies in $[L_a,U_a]$. For each output component $b$ and address $x_j$, sum the smaller and the larger of $L_aQ_{j,ab}(x_j)$ and $U_aQ_{j,ab}(x_j)$ over $a$. Minimize the lower sums and maximize the upper sums over $x_j$ to enclose $y_{j,b}$ for every joint prefix. Applying floor division to both endpoints encloses $z_{j,b}$. Initialization uses $[1,1]$. Independence between components is unnecessary because interval sums enclose even correlated operands. A width $w\ge1$ is sufficient exactly when $-2^{w-1}\le L\le U\le2^{w-1}-1$ for the chosen enclosing interval. The implementation chooses the least such width for each scalar interval. Coefficient words are independently enclosed over the corresponding coordinate table.

All arithmetic is modular until an accumulator is interpreted. The signed shift-and-add multiplier sign-extends the multiplicand and subtracts the sign-bit partial product of a quantum coefficient. Terms above the accumulator width vanish modulo $2^w$. A partial addition may wrap, but its residue contributes the correct modular sum. The final exact signed sum lies in the proven raw interval, so its represented two's-complement value equals the mathematical dot product. Arithmetic floor uses the high bits of this signed word, including sign extension when the shift consumes all its non-sign bits. The retained signed interval makes the smaller view sufficient.

The first stage can precompute floor division in its coordinate-local lookup. A later stage can retain the raw dot and expose only the retained bits, or compute a raw dot in a shared register, XOR the retained signed bits into a clean output, and invert the dot computation. The second strategy spends additional gates to reuse raw-dot storage. The coefficient bank and adder work are unloaded or restored. For either schedule, write the entire forward computation as a reversible permutation $C$ on addresses and clean workspace. A word-dependent $P$ rotation acts diagonally on its computed basis image. Consequently $C^\dagger P C|x,0\rangle=e^{i\theta(x)}|x,0\rangle$ for every address, including arbitrary coherent superpositions by linearity. This argument requires no inverse of a TT matrix and no reversible overwrite of a singular or rectangular core.

## Componentwise approximation

For every prefix, let $M_a$ bound the absolute component of the unquantized prefix, $C_a$ bound its difference from the unrounded quantized prefix, and $R_a$ bound the difference between the unrounded quantized prefix and the rounded one. Initially $(M,C,R)=(1,0,0)$. Update each output bound by maximizing the following expressions over the **same** coordinate slice $x_j$:

$$\begin{aligned}
M'_b &= \max_{x_j}\sum_a M_a |A_{j,ab}|,\\
C'_b &= \max_{x_j}\sum_a \big[C_a|B_{j,ab}|+M_a|A_{j,ab}-B_{j,ab}|\big],\\
R'_b &= \max_{x_j}\left(\sum_a R_a|B_{j,ab}|+\rho_{j,b}(x_j)\right).
\end{aligned}$$

Indeed, if the exact and unrounded quantized prefixes are $p$ and $q$, then $pA_j-qB_j=(p-q)B_j+p(A_j-B_j)$. Subtracting the rounded prefix update from $qB_j$ gives the third inequality. The local rounding defect is an integer remainder divided by $2^{g_j}$, so $0\le\rho\le(2^{k_j}-1)2^{-g_j}$. The actual remainder is known at the first lookup. These estimates can be computed exactly for dyadic data. A simpler sufficient policy uses $|A_{j,ab}-B_{j,ab}|\le2^{-f_j-1}$ and $\rho<2^{-s_j}$ for stages with nonzero shift. The final scalar bound is $\|A-\widehat V\|_\infty\le C_d+R_d$. It may overestimate error because it ignores cancellation and correlations between earlier coordinates. An exact grid residual and exact emitted-angle comparison can certify additional candidates on small grids.

An invertible bond gauge $G$ preserves $A_jG\,G^{-1}A_{j+1}$ in exact arithmetic but can change all three recurrences, coefficient widths and QROM structure. Power-of-two diagonal gauges keep dyadic products exact if each transformed entry is representable. Reversing a chain and transposing the slices also changes the sequence of range and error amplification. Neither canonicalization nor one contraction order is uniformly better. These policies certify the particular supplied TT, direction, and rounding schedule.

## From a signed word to a phase

For represented $\beta$, an ideal signed word contributes angles $\theta_i=\beta 2^{-s_d}w_i$, where $w_i=2^i$ except for the negative sign weight. If each stored angle is $\bar\theta_i$, the worst parameter-encoding error is at most $E_{\rm enc}=\sum_i|\theta_i-\bar\theta_i|$ by the triangle inequality. This sum is evaluated as a rational number. With any operator-norm budget $\eta$ for subsequent rotation synthesis in a consistent phase convention, the clean-input isometry error satisfies

$$\|U-\widehat U_{\rm synth}\|_{\rm clean}\le
\min\{2,\,|\beta|(\tau+C_d+R_d)+E_{\rm enc}+\eta\}.$$

For the native clean diagonal this uses $|e^{ia}-e^{ib}|\le\min(2,|a-b|)$ and a maximum over addresses. Replacing gates by synthesized approximations is bounded on the entire joint circuit by unitary telescoping, even if synthesis introduces leakage outside the clean subspace. The stated error target is guaranteed when the right side is at most $\varepsilon$ and every interpreted word fits its signed interval. T counts in this repository are projections. The synthesis hypothesis is separate from the native arithmetic certificate.

The independent finite-grid check executes exact Boolean paths through X, CX and CCX, sums the represented angles of P and Rz exactly, and checks identity action on all addresses and zero workspace at exit. For an angle difference $a$, any integer $m$ and an enclosure $[l,u]$ of $2\pi$ give an upper bound $\max(|a-ml|,|a-mu|)$ on $|a-2m\pi|$. The code obtains an enclosure by the alternating series in Machin's identity. This certifies emitted finite circuits including angle encoding, and does not depend on rounded evaluations of sine or cosine. It costs at least the number of grid points and is a validation method for the measured grids. The range and componentwise bounds are coordinate-local alternatives.

## Composition, synthesis, and probability

For one midpoint let $P=e^{i\beta V}$ and let $K_+$ denote the positive **kinetic unitary**. In chronological order the four potential calls are $P,P^\dagger,P^\dagger,P$. The complete matrix is

$$W(P)=(P K_+ P^\dagger)K_+^\dagger(P^\dagger K_+^\dagger P)K_+.$$

A global factor on $P$ cancels identically when inverses are adjoints. If $\delta=\|P^\dagger\widetilde P-cI\|$ for any unit complex $c$, then $\|W(P)-W(\widetilde P)\|\le4\delta$. The measured schedule minimizes this certified expression over the unshifted comparison and a center of the exact address angle differences. A sharpening is available for a known kinetic block. Set $E=\bar cP^\dagger\widetilde P$. The first conjugation difference has norm $\|[E,K_+]\|$, bounded by $2\delta\min(1,\|K_+-zI\|)$ for unit complex $z$, by subtracting $I$ from $E$ and either zero or $zI$ from $K_+$. The second conjugation obeys the same bound using the conjugate kinetic phase and unitary similarity. Telescoping between the two conjugations gives $4\delta\min(1,\|K_+-zI\|)$. The recorded certificate uses the simpler factor 4 for the common native kinetic circuit.

Sum the per-step bounds through the unitary schedule. Add any independent source, kinetic and total synthesis bounds required by the reference being compared. In the experiment the reference uses the same compiled native source and kinetic factors, with the exact phase of each recorded binary64 potential. Thus those two common factors introduce no **difference in this comparison**. At the end the output distribution TV is at most the trace distance of the prepared states, which for normalized vectors is at most their vector 2-norm difference. This also bounds any fixed event bias. At depth $k$, amplification uses $2k+1$ occurrences of the preparation or its inverse. The corresponding fixed-reference circuit and event errors are bounded by multiplying the native preparation bound by $2k+1$ and capping the event bound at one. Repeated shots provide statistical information about the implemented distribution, with no removal of its fixed model or preparation bias.

To compare with a continuously time-varying finite-grid Hamiltonian $i[K_N,V(t)]$, time freezing at midpoint $t_m$ contributes at most $\|K_N\|L_t h^2/2$ in a step if $\|V(t)-V(t_m)\|_\infty\le L_t|t-t_m|$. This follows by Duhamel, $\|[K_N,D]\|\le 2\|K_N\|\|D\|$, and the integral of $|t-t_m|$. The commutator product formula incurs a separate constant-potential local error of order $h^2$ under Layden et al.'s Theorem 2, for the theorem's scaling and norm conditions. The numerical experiment checks time refinement and reports finite-grid versus continuum and learned versus target differences separately. Its exact certificate is for the implemented finite-grid preparation relative to the stated native-block reference. The experiment supplies no interval enclosure of the continuum solution or numerical upper bound on the theorem's learned-model regularity constants. [Layden et al., Theorem 2 and Appendix C](https://arxiv.org/html/2510.08462v1#A3)

For the benchmark's balanced scales, a sufficient frozen-step bound also follows directly without identifying those scales with the paper's Eq. (37). Let $A=i\beta V_m$, $B=i\alpha K_N$ for a frozen real diagonal $V_m$, and $C=[K_N,V_m]$. Replace each occurrence of $A,B$ in the eight-factor formula by $sA,sB$. The two group commutators give $I+s^2[A,B]+s^3D+O(s^4)$ and $I+s^2[A,B]-s^3D+O(s^4)$ for the same cubic term $D$. Hence the Taylor polynomial through degree three is $I+2\alpha\beta s^2 C$. The norm of the fourth derivative on $0\le s\le1$ is bounded by $(4\|A\|+4\|B\|)^4$ by Leibniz's rule and unitarity. The Taylor remainder for $e^{hC}$ through its linear term has norm at most $h^2\|C\|^2/2$ by a double integral and unitarity. The frozen-step error is therefore bounded by

$$\min\!\left\{2,\frac{(4\|A\|+4\|B\|)^4}{24}+\frac{h^2\|C\|^2}{2}+|2\alpha\beta-h|\|C\|\right\}.$$

The norms of $K_N,V_m$ in this bound can be taken after independent scalar centering because the commutator and the eight-factor product both remove scalar offsets. With $\alpha,\beta=O(\sqrt h)$ and $2\alpha\beta=h$, the first two terms are $O(h^2)$. This bound is conservative and distinct from the measured same-grid ODE error. A comparison against the analytic kinetic unitary and the exact source would also require bounds on their native compilations and the error in materializing the frozen table.
