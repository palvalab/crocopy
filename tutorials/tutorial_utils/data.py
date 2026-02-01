import os.path as op

import mne
from mne.minimum_norm import apply_inverse_epochs, make_inverse_operator
from mne.preprocessing import compute_proj_ecg, compute_proj_eog
from mne.minimum_norm import apply_inverse_raw

# data from MNE
# https://mne.tools/mne-connectivity/stable/auto_examples/mne_inverse_envelope_correlation.html
def load_example_data(max_time=60):
    data_path = mne.datasets.brainstorm.bst_resting.data_path()
    subjects_dir = op.join(data_path, "subjects")
    subject = "bst_resting"
    trans = op.join(data_path, "MEG", "bst_resting", "bst_resting-trans.fif")
    src = op.join(subjects_dir, subject, "bem", subject + "-oct-6-src.fif")
    bem = op.join(subjects_dir, subject, "bem", subject + "-5120-bem-sol.fif")
    raw_fname = op.join(
        data_path, "MEG", "bst_resting", "subj002_spontaneous_20111102_01_AUX.ds"
    )

    raw = mne.io.read_raw_ctf(raw_fname, verbose="error")
    raw.crop(0, max_time).pick_types(meg=True, eeg=False, verbose="error").load_data(verbose="error").resample(500)
    raw.apply_gradient_compensation(3)
    projs_ecg, _ = compute_proj_ecg(raw, n_grad=1, n_mag=2, verbose="error")
    projs_eog, _ = compute_proj_eog(raw, n_grad=1, n_mag=2, ch_name="MLT31-4407", verbose="error")
    raw.add_proj(projs_ecg + projs_eog, verbose="error")
    raw.apply_proj(verbose="error")
    raw.filter(0.1, None, verbose="error")  # this helps with symmetric orthogonalization later
    cov = mne.compute_raw_covariance(raw, verbose="error")  # compute before band-pass of interest

    src = mne.read_source_spaces(src, verbose="error")
    fwd = mne.make_forward_solution(raw.info, trans, src, bem, verbose="error")
    del src
    inv = make_inverse_operator(raw.info, fwd, cov, verbose="error")
    del fwd

    labels = mne.read_labels_from_annot(subject, "aparc_sub", subjects_dir=subjects_dir, verbose="error")

    lambda2 = 1.0 / 9.0 
    method = "dSPM"  

    stc_raw = apply_inverse_raw(raw, inv, lambda2=lambda2, method=method,  pick_ori="normal" , verbose="error")
    data_broadband = mne.extract_label_time_course(
        stc_raw, labels, src=inv["src"],  
        mode="mean_flip",                
        return_generator=False,
        verbose="error"
    ) 

    label_names = [lab.name for lab in labels]

    return data_broadband, label_names, raw.info

