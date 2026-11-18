"""Future SAM 3 training workspace for an OCEAN AI company installation.

This module prepares a local training/deployment plan. It deliberately does not
run training or send images to a hosted model. Existing Roboflow inference is
managed separately by ocean_three_views.py and roboflow_client.py.
"""
import json
from datetime import datetime, timezone

import pandas as pd
import streamlit as st


def _step(number, title, detail):
    st.markdown(
        f'<div class="ocean-lab-step">'
        f'<span class="ocean-step-number">{number}</span>'
        f'<div><strong>{title}</strong>'
        f'<small>{detail}</small></div></div>',
        unsafe_allow_html=True,
    )


def show_training_lab(snapshot):
    imported, records = snapshot()
    st.markdown('''
    <div class="ocean-page-intro">
      <span class="ocean-eyebrow">MODEL DEVELOPMENT / FUTURE LOCAL WORKSTATION</span>
      <h1>SAM 3 Training Lab</h1>
      <p>Prepare datasets, plan fine-tuning, and get ready for an eventual
      installation on the company's own computer. The live cloud detection
      system stays unchanged.</p>
    </div>
    ''', unsafe_allow_html=True)

    a, b, c = st.columns(3)
    with a:
        st.metric('Imported inspection images', f'{imported:,}', help='Images available to the current app installation; not necessarily training-ready.')
    with b:
        st.metric('Saved AI results', f'{len(records):,}', help='Predictions are not verified training labels.')
    with c:
        st.metric('Local training engine', 'Not connected', help='No training service is installed or started by this page.')

    st.info('**No training is running.** SAM 3 already works zero-shot through Roboflow. Local inference and optional fine-tuning can be set up later on the company workstation.')

    tab1, tab2, tab3, tab4 = st.tabs([
        'Preparation', 'Dataset readiness', 'Local training plan', 'Company deployment'
    ])

    with tab1:
        left, right = st.columns([1.15, 1], gap='large')
        with left:
            st.subheader('When should we fine-tune SAM 3?')
            st.write('Start with the **pretrained SAM 3 model** and test it on representative, engineer-reviewed images. Fine-tuning is worth considering only if the baseline keeps missing real corrosion or highlighting clean surfaces.')
            st.warning('The earlier 14-image evaluation showed broad over-segmentation. A cleaner visualization does **not** mean the model became more accurate.')
            st.markdown('**Two separate future tasks**')
            st.markdown('**Local inference:** install pretrained SAM 3 so inspections can run on company hardware, subject to compatibility testing.\n\n**Optional fine-tuning:** adapt it using corrected training masks, then validate it on held-out images.')
        with right:
            st.subheader('Recommended workflow')
            _step('01', 'Collect and label', 'Use original images with engineer-corrected corrosion masks.')
            _step('02', 'Create fixed splits', 'Keep training, validation, and untouched test scenes separate.')
            _step('03', 'Measure the baseline', 'Track pixel IoU, precision, recall, and false/missed detections.')
            _step('04', 'Fine-tune if needed', 'Train on an approved GPU workstation, not the hosted dashboard.')
            _step('05', 'Promote only if better', 'Save the checkpoint and compare it with the baseline on the same test set.')

    with tab2:
        st.subheader('Dataset readiness')
        st.caption('Counts below reflect this installation only. Predicted masks must not be treated as ground-truth annotations.')
        x, y = st.columns([1.1, 1], gap='large')
        with x:
            if records:
                table = pd.DataFrame([
                    {
                        'Inspection image': r['name'],
                        'AI candidates': r['regions'],
                        'Human decision': r['review'],
                    }
                    for r in records[:30]
                ])
                st.dataframe(table, hide_index=True, use_container_width=True, height=300)
                if len(records) > 30:
                    st.caption(f'Showing the newest 30 of {len(records)} saved results.')
            else:
                st.info('No inspection results saved in this installation yet. You can still prepare the future training workflow.')
        with y:
            st.markdown('**Before a dataset can be used for training**')
            st.markdown('''
- [ ] Original images have permission for company processing
- [ ] Corrosion masks are reviewed and corrected by a qualified reviewer
- [ ] Clean / negative examples are included
- [ ] Duplicate frames and near-identical scenes are grouped
- [ ] Train, validation, and test splits avoid scene leakage
- [ ] Dataset version, image source, and label definitions are recorded
''')
            st.caption('Review decisions in the Results Library are not equivalent to pixel-level mask approval. A verified annotation registry is a future feature.')

    with tab3:
        st.subheader('Prepare a company-local configuration')
        st.caption('This exports a planning file, not a runnable SAM 3 training command. It does not start a job or access your company PC.')
        col1, col2 = st.columns(2, gap='large')
        with col1:
            goal = st.selectbox(
                'Planned operation',
                ['Local inference with pretrained SAM 3', 'Fine-tune SAM 3 after masks are verified'],
                key='sam3_lab_goal',
            )
            compute = st.selectbox(
                'Target computer',
                ['Company GPU workstation', 'Company GPU server', 'To be decided'],
                key='sam3_lab_compute',
            )
            dataset_name = st.text_input(
                'Dataset/version label', value='corrosion_verified_v1',
                key='sam3_lab_dataset_name',
                help='A proposed label only. Nothing is uploaded or created here.',
            )
        with col2:
            model_label = st.text_input(
                'Model identifier', value='SAM 3 pretrained baseline',
                key='sam3_lab_model_name',
            )
            hardware_notes = st.text_area(
                'Company PC / GPU notes',
                placeholder='GPU model, VRAM, operating system, storage, and available drivers',
                height=118, key='sam3_lab_hardware',
            )
        fine_tune = goal.startswith('Fine-tune')
        if fine_tune:
            st.warning('Fine-tuning needs a reviewed segmentation dataset and an approved, tested local training environment. The model weights, dependencies, and training script must be installed separately.')
        else:
            st.success('Good first step: run pretrained SAM 3 locally and compare its outputs with your cloud baseline before deciding whether fine-tuning is necessary.')

        plan = {
            'project': 'OCEAN AI',
            'document_type': 'future_local_sam3_plan',
            'generated_utc': datetime.now(timezone.utc).isoformat(),
            'operation': 'fine_tuning' if fine_tune else 'pretrained_inference',
            'execution_environment': compute,
            'dataset_label': dataset_name.strip(),
            'model_label': model_label.strip(),
            'hardware_notes': hardware_notes.strip(),
            'requires_verified_masks': fine_tune,
            'training_engine_connected': False,
            'deployment_status': 'planned',
            'warning': 'Planning record only. This file cannot start training. Verify hardware, model license, and dataset permissions before deployment.',
        }
        st.download_button(
            'Download local SAM 3 plan (.json)',
            data=json.dumps(plan, indent=2),
            file_name='ocean_ai_sam3_local_plan.json',
            mime='application/json',
            type='primary',
            use_container_width=False,
            key='sam3_lab_export',
        )
        with st.expander('Preview exported plan'):
            st.json(plan)

        st.markdown('**Training runs & checkpoints**')
        st.caption('There is no connected local training agent yet, so this section cannot launch jobs or report metrics. Run history, checkpoints, and job controls will be added when the company machine is ready.')

    with tab4:
        st.subheader('Future installation architecture')
        _step('01', 'Company computer', 'Confirm GPU, VRAM, drivers, operating system, storage, and permitted model use.')
        _step('02', 'Install local inference', 'Set up SAM 3 weights and dependencies on the company workstation/server.')
        _step('03', 'Connect the dashboard', 'Integrate a local inference/training worker without changing the cloud workflow.')
        _step('04', 'Secure inspection records', 'Add persistent storage, user access controls, backups, and engineer review history.')
        _step('05', 'Validate before release', 'Compare local masks to reviewed ground truth and approve a model version.')
        st.info('Heavy inference and optional training should run on the company workstation/server, not on the ROV Raspberry Pi 4 or Streamlit Community Cloud.')
        st.link_button('Official Meta SAM 3 repository', 'https://github.com/facebookresearch/sam3')
