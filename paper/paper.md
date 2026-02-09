---
title: >-
	CROCpy - A Python toolbox for the analysis of  CRitical Oscillations and Connectivity
authors:
  - name: Vladislav Myrov
    affiliation: "1"
  - name: Felix Siebenhühner
    affiliation: "1, 2"
  - name: Sheng H Wang
    affiliation: "1, 2, 3"
  - name: Gabriele Arnulfo
    affiliation: "4, 6"
  - name: Joonas J. Juvonen
    affiliation: "1"
  - name: Monica Roascio
    affiliation: "4"
  - name: Alina Suleimanova
    affiliation: "1"
  - name: Joona Repo
    affiliation: "1"
  - name: Wenya Liu
    affiliation: "1, 2"
  - name: Satu Palva
    affiliation: "2, 5"
  - name: J. Matias Palva
    affiliation: "1, 2, 5"
affiliations:
  - index: 1
    name: Department of Neuroscience and Bioengineering (NBE), Aalto University, Espoo, Finland
  - index: 2
    name: Neuroscience Center, Helsinki Institute of Life Science, University of Helsinki, Finland
  - index: 3
    name: Jack H. Miller MEG Center, Helen DeVos Children’s Hospital, Corewell Health, USA
  - index: 4
    name: UNIGE, University of Genoa, Genoa, Italy
  - index: 5
    name: Centre for Cognitive Neuroimaging, Institute of Neuroscience and Psychology, University of Glasgow, UK
  - index: 6
	name: IRCCS Giannina Gaslini, Genoa, Italy
bibliography: crocpy_refs.bib
tags:
  - electrophysiology
  - oscillations
  - synchronization
  - critical dynamics
---

# Summary

`CROCpy` is a light-weighted toolbox for the assessment of neuronal oscillations, and multiple observables of functional connectivity (phase synchronization, amplitude coupling, and cross-frequency coupling) and critical dynamics (avalanches, long-range temporal correlations, bistability, and functional excitation-inhibition ratio). It was developed to simplify the analysis of continuous electrophysiological recordings and, in addition to metric computation, also includes methods for narrow-band filtering and statistical analysis. It is device-agnostic and supports both GPU and CPU computations. The toolbox also provides detailed tutorials.

# Statement of need

The brain is a complex dynamical system, where neuronal interactions across spatiotemporal scales give rise to neural oscillations. These oscillations are related to each other by various forms of functional connectivity, and exhibit various forms of scale invariance such as avalanche dynamics, long-range temporal correlations (LRTCs), and bistability. Such concurrent and diverse activity patterns are hypothesized to emerge when a system is operating near a critical transition between order and disorder, a conceptual framework known as *brain criticality*.

Increasing evidence suggests that functional connectivity and critical dynamics are closely related to each other. Studying these together can therefore offer a holistic picture of neuronal dynamics. We here present `CROCpy`, a unified and open-source Python toolbox, which combines implementations of multiple metrics used to analyze neuronal oscillations, their functional connectivity, and (critical) dynamics. The metrics it provides can be used for both basic research and as biomarkers in clinical contexts.

# State of the field

While several toolboxes exist that offer implementations of metrics for either criticality or connectivity, to our knowledge, none of these include the existing metrics used in studies of critical dynamics and connectivity altogether, focusing on specific methods such as DFA exponent or connectivity. Importantly, most existing toolboxes are not optimized for speed and do not utilize GPU which limits their applicability to large datasets.

# Software design

`CROCpy` was designed to be as simple and flexible as possible and operates with basic N-dimensional arrays. Many operations can be indiscriminately run on either CPUs or GPUs, flexibly using either numpy or cupy under the hood.  For most metrics, `CROCpy` supports GPU acceleration for efficient analysis of very large datasets. Additionally, `CROCpy` implements routines for narrow-band filtering and several commonly used statistical tests, with tutorials provided on GitHub.

Observables can be grouped in three categories (Table 1):

|             Category            |                Construct                |                       Observables                       |                 Citation                 |
|:-------------------------------:|:---------------------------------------:|:-------------------------------------------------------:|:----------------------------------------:|
|           Oscillations          |          Phase autocorrelation          |                        pAC value                        |                [@Myrov2024]              |
|                                 |                                         |                      pAC lifetimes                      |                                          |
|           Criticality           |                Avalanches               |                      Avalanche size                     | [@Palva2013; @Zhigalov2015]              |
|                                 |                                         |                    Avalanche duration                   |                                          |
|                                 |                  LRTCs                  |                   DFA scaling exponent                  | [@LinkenkaerHansen2001]                  |
|                                 | Functional  excitation-inhibition ratio |                           fE/I                          |               [@Bruining2020]            |
|                                 |               Bistability               |                    Bistability index                    |  [@Freyer2009; @Wang2023]                |
| Oscillation-based  connectivity |             Phase Synchrony             |                   Phase locking value (PLV)                   |  [@Palva2012]                            |
|                                 |                                         |         Imag. component of  PLV         |                                          |
|                                 |                                         |                 Weighted phase lag index (wPLI)                |   [@Vinck2011]                           |
|                                 |            Amplitude Coupling           |                  Amplitude correlation                  |                                          |
|                                 |                                         |         Amplitude correlation  (orthogonalized)         |               [@Brookes2012]             |
|                                 |        Cross-frequency  Coupling        | Cross-frequency phase coupling |   [@Palva2018CF; @Siebenhhner2020]         |
|                                 |                                         |                 Phase-amplitude coupling                |                   [@Palva2018CF]           |
|                                 |                                         |             Filtered amplitude correlations             |             [@Siebenhhner2020]           |

Brief description could be found in the next section while detailed background, usage tips and interpretation could be found in Tutorials.

# Implementation and the core methods

## Phase autocorrelations

A neural oscillation “order” quantifies the local synchronization. Although its amplitude is a proxy for synchronization, it is not bounded to a specific interval and depends on other parameters. The phase autocorrelation function (pACF) quantifies the rhythmicity of a narrow-band signal by computing the self-similarity of its phase with a time-lagged version. The pACF operationalizes rhythmicity of oscillations either by the average value within a given lag range or by their lifetime (the lag at which the pACF crosses a specified threshold) [@Myrov2024].

## Criticality

In the _criticality_ framework, the brain is hypothesized to be comparable to a system operating near a phase transition point, where its dynamics become scale-free and correlations extend across a wide range of temporal and spatial scales. Scale-free behavior indicates that a process has no single “typical” temporal or spatial scale that dominates its dynamics and is characterized by power-law scaling of relevant statistical quantities over orders of magnitude. 

### Neuronal Avalanches

![**A** Suprathreshold peak detection and time binning. **B** Peak counts per time bin. **C** Avalanche duration distribution with power-law fit.](figures/avalanches_panel.png){#fig:avalanches style="width: 70%; margin: auto; align: right;" }

Neuronal avalanches in multichannel MEG/SEEG recordings are brief cascades of large-amplitude neuronal events that propagate across brain areas and exhibit power-law event size and lifetime distributions. `CROCpy` first detects in each channel the peaks (local maxima) whose amplitude exceeds a chosen threshold $T$ (Figure 1A), convert each suprathreshold peak to a binary event, after which the time series is discretized into regularly-spaced time bins within which the number of spiking events are counted (Figure 1B). An avalanche is defined as a contiguous sequence of non-empty bins and its size is quantify as the total number of suprathreshold peaks in that sequence and its duration is the number of consecutive occupied bins [@Zhigalov2015]. Finally, the power-law is fitted to a distribution of avalanche sizes/lengths (Figure 1C).

### Long range temporal correlations (LRTCs)

![**A** Narrowband envelope time series. **B** Integrated profile with linear detrending in each window. **C** Fluctuation as a function of window size with linear fit (DFA exponent).](figures/dfa_panel.png){#fig:dfa style="width: 70%; margin: auto; align: right;"}

LRTCs in narrow-band oscillations describe how the amplitude of an oscillation fluctuates in a structured way over time. It can be assessed with detrended fluctuation analysis (DFA; [@LinkenkaerHansen2001]). First, the signal profile is obtained as  the cumulative sum of demeaned amplitudes, and detrended fluctuation analysis (DFA) is performed using logarithmically increasing window sizes (Figure 2A). For each window, a linear trend is fitted and removed, and the detrended fluctuation is quantified as the root-mean-square deviation from the fitted trend for that window size (Figure 2B). Finally, the DFA scaling exponent is obtained as the slope of a linear regression between log window size and log fluctuation. (figure 2C). This DFA procedure allows for the exclusion of "bad" samples (e.g., high-amplitude artifacts) but is slow to compute. We also implemented a spectral-domain approach to estimate signal variability [@Nolte2019], which could be ten times faster but does not support artifact rejection.

### Neuronal Bistability inside a Critical Regime

![**A** Narrowband envelope time series. **B** Envelope power normalized to 0-1 range. **C** Power distribution with exponential and bi-exponential fits.](figures/bis_panel.png){#fig:bis style="width: 70%; margin: auto; align: right;"}

Bistability refers to dynamics that alternate between “down” and “up” states of synchrony, arising from positive feedback mechanisms, and is nonlinearly related to measures of criticality [@Wang2023] (Figure 3A). First, narrow-band power time-series (squared envelope) is computed and normalized by maximum value (Figure 3B). Next, 'CROCpy' compares whether its empirical distribution is better fitted by single exponential or bi-exponential function using Bayesian Information Criterion (BIC, Figure 3C, [@Freyer2009]).

$$
\mathrm{BiS}=
\begin{cases}
\log_{10}(\Delta \mathrm{BIC}), & \Delta \mathrm{BIC}>0,\\
0, & \Delta \mathrm{BIC}\le 0~.
\end{cases}
\label{eq:bis}
$$

Thus, BIS $>$ 0 indicates that the bi-exponential model (two-state structure) explains the power fluctuations better.

### Functional Excitation-Inhibition ratio (fE/I)

![ **A** Narrowband envelope time series. *B* Integrated profile normalized by the window standard deviation. *C* Correlation across windows between mean amplitude and normalized fluctuation.](figures/fei_panel.png){#fig:fei style="width: 70%; margin: auto; align: right;"}

The fE/I index is intended to infer the local E/I ratio from a time series by quantifying how a critical oscillation (e.g., with DFA $>$ 0.6) co-varies with the short-timescale temporal structure of amplitude fluctuations [@Bruining2020]. Like DFA analysis, the narrow-band signal profile is obtained (Figure 4A) and epoched into overlapping windows, and the average envelope is computed for each window ($W_{env}$) (Figure 4B). Next, each window is amplitude-normalized by the corresponding mean envelope, detrended, and the fluctuation within the window is defined as the standard deviation ($W_{fluct}$). The fE/I ratio is defined as

$$
fE/I = 1 - \rho(W_{env}, W_{fluct}),
\label{eq:fei}
$$

where $\rho\!\left(W_{\mathrm{env}}, W_{\mathrm{fluct}}\right)$ is the Pearson correlation across windows between their envelope and fluctuation (Figure 4C). The $fE/I$ is bounded between 0 and 2 by construction: $fE/I<1$ indicates inhibition-dominated (subcritical) dynamics, $fE/I>1$ excitation-dominated (supercritical) dynamics, and $fE/I\approx 1$ a near-critical balanced regime.

## Functional connectivity (FC)

FC denotes the statistical dependencies between neurophysiological signals recorded from different neuronal populations and is interpreted as inter-areal functional relationships that support communication and information transfer [@Fries2015; @Palva2012]. FC in electrophysiological data comes in several types and can be estimated across a wide range of frequencies.


### Phase synchrony (PS)

![**A** Narrowband signals $X$ and $Y$, their phases, and phase difference $\Delta\theta(t)$. **B** Phase-difference distribution (circular plot) and corresponding imaginary-part distributions used for PLV/iPLV/wPLI.](figures/ps_panel.png){#fig:ps width="100%"}

PS refers to coupling between brain regions via a temporally stable phase difference between narrow-band signals (Figure 6A, [@Palva2012]). With 'CROCpy', PS can be estimated using the phase locking value (PLV), the imaginary phase-locking value (iPLV), and weighted Phase Lag Index (wPLI). EEG/MEG data are affected by linear signal mixing, which induces spurious interactions at zero phase lag [@Palva2018Ghost]. To avoid these false positives, iPLV only uses the imaginary component of the complex phase lag (which is zero when the phase lag is zero or a multiple of pi, Figure 6B), while the wPLI down-weights phase relationships near zero lag by weighting the phase difference by the magnitude of the imaginary component (Figure 6C, [@Vinck2011]).

Both PLV and iPLV can be derived from the complex-valued PLV which is defined as: 

$cPLV = \frac{1}{N} \sum_{i=1}^N x_n*y_n^*$

where $X_n$ is unit-normed complex signal X ($X_n = \frac{X_i}{|X_i|}$) and $Y_n^*$  is conjugate of unit-normed complex signal Y. This yields a complex value, from which the regular PLV ($|cPLV|$) and the iPLV ($|img(cPLV)|$) can be obtained. 

### Amplitude correlations (AC)

![**A** Narrowband amplitude signals and their envelopes $X$ and $Y$, and $Y_{\perp X}$ (orthogonalized w.r.t.\ $X$). **B** Amplitude correlation for $X$ vs.\ $Y$ (left) and $X$ vs.\ $Y_{\perp X}$ (right).](figures/cc_figure.png){#fig:ac width="100%"}

AC reflects inter-areal coupling of local fluctuations in neuronal firing patterns. Most simply, AC can be assessed by computing the Pearson correlation coefficient (CC) between envelopes of complex time series from two brain areas. However, CC is inflated by zero-lag artificial interactions as is the case for PS. To address this, the orthogonalized CC (oCC) was developed [@Brookes2012], in which the correlation is computed between signal X and Y orthogonalized with respect to X ($Y \perp X$) as (Figure 6A,B):

$$
y_{\perp x}= i*x_n*imag( y(t)*x_n^*)
$$

## Cross-frequency coupling

![**A** Narrowband signals $X$ and $Y$, their phases, and the phase of $X$ multiplied on $\frac{m}{n}, (m=2, n=1)$ $Y$. **B** Narrowband signals $X$ and $Y$ and the amplitude envelope of $X$.](figures/cfc_panel.png){#fig:cfc width="100%"}

Various studies have indicated that neuronal oscillations interact not only within frequency bands, but also through multiple forms of cross-frequency coupling (CFC). These can be computed between different frequencies for the same region (local CFC) or between regions (long-range CFC) [@Siebenhhner2020]. Two widely investigated forms of CFC are cross-frequency phase synchrony (CFS) and phase-amplitude coupling (PAC).  

CFS, also known as phase-phase coupling, is an extension of PLV to two oscillations with an integer frequency ratio n:m. CFS is estimated by estimating the phase-locking  after phase-accelerating the slower oscillation (i.e., by multiplying its phase by m/n, Figure 7A). As CFC is assumed to be largely unaffected by linear mixing, it is not necessary to discard the real component of the cPLV [@Palva2018CF].

PAC captures the dependence between the phase of a slower rhythm and the amplitude envelope of a faster rhythm, testing whether high-frequency power is systematically modulated at specific phases of the low-frequency oscillation. Several approaches exist for estimating PAC that generally yield comparable results. 'CROCpy' implements PLV-PAC, which estimates PAC as the PLV between the slow oscillations' phase and the phase of the amplitude envelope of the fast oscillation after wavelet convolution with the slow frequency (Figure 7B).

## Outlook and Future Efforts

The tools described above reflect the current implementation as of January 2026. We are a group of enthusiastic researchers actively developing novel criticality-inspired biomarkers, investigating their mechanistic links with established neurophysiological measures, and optimizing their use for studying complex neuroscience problems. Accordingly, this toolbox will be actively maintained, with ongoing improvements and the addition of new tools made available through GitHub. We warmly welcome contributions from researchers with shared interests and invite the community to participate in this joint, long-term effort.

## Research impact statement

CROCpy has been used in multiple publications from several labs, including analyses of resting-state iSEEG functional connectivity of high-frequency oscillations, the relationship between DFA and phase synchronization, MEG/EEG data from patients with epilepsy, time-resolved features of multi-day ECoG recordings, and changes in brain dynamics during sleep.

# AI usage disclosure

AI (ChatGPT 5.2) was used to proof-reading the manuscript.

# References
