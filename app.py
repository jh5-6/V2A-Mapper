import gradio as gr
import numpy as np
from audioldm import image_to_audio, build_model

model_id = "haoheliu/AudioLDM-S-Full"
audioldm = None
current_model_name = None

def image2audio(image, duration, guidance_scale, random_seed, n_candidates, model_name):
    global audioldm, current_model_name
    
    if audioldm is None or model_name != current_model_name:
        audioldm=build_model(model_name=model_name)
        current_model_name = model_name
        
    waveform = image_to_audio(
        latent_diffusion=audioldm,
        images=[image],
        seed=random_seed,
        duration=duration,
        guidance_scale=guidance_scale,
        n_candidate_gen_per_text=int(n_candidates),
    )  # [bs, 1, samples]

    waveform = [
        gr.make_waveform((16000, wave[0]), bg_image="bg.png") for wave in waveform
    ]
    # waveform = [(16000, np.random.randn(16000)), (16000, np.random.randn(16000))]
    if len(waveform) == 1:
        waveform = waveform[0]
    return waveform

css = """
        a {
            color: inherit;
            text-decoration: underline;
        }
        .gradio-container {
            font-family: 'IBM Plex Sans', sans-serif;
        }
        .gr-button {
            color: white;
            border-color: #000000;
            background: #000000;
        }
        input[type='range'] {
            accent-color: #000000;
        }
        .dark input[type='range'] {
            accent-color: #dfdfdf;
        }
        .container {
            max-width: 730px;
            margin: auto;
            padding-top: 1.5rem;
        }
        #gallery {
            min-height: 22rem;
            margin-bottom: 15px;
            margin-left: auto;
            margin-right: auto;
            border-bottom-right-radius: .5rem !important;
            border-bottom-left-radius: .5rem !important;
        }
        #gallery>div>.h-full {
            min-height: 20rem;
        }
        .details:hover {
            text-decoration: underline;
        }
        .gr-button {
            white-space: nowrap;
        }
        .gr-button:focus {
            border-color: rgb(147 197 253 / var(--tw-border-opacity));
            outline: none;
            box-shadow: var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow, 0 0 #0000);
            --tw-border-opacity: 1;
            --tw-ring-offset-shadow: var(--tw-ring-inset) 0 0 0 var(--tw-ring-offset-width) var(--tw-ring-offset-color);
            --tw-ring-shadow: var(--tw-ring-inset) 0 0 0 calc(3px var(--tw-ring-offset-width)) var(--tw-ring-color);
            --tw-ring-color: rgb(191 219 254 / var(--tw-ring-opacity));
            --tw-ring-opacity: .5;
        }
        #advanced-btn {
            font-size: .7rem !important;
            line-height: 19px;
            margin-top: 12px;
            margin-bottom: 12px;
            padding: 2px 8px;
            border-radius: 14px !important;
        }
        #advanced-options {
            margin-bottom: 20px;
        }
        .footer {
            margin-bottom: 45px;
            margin-top: 35px;
            text-align: center;
            border-bottom: 1px solid #e5e5e5;
        }
        .footer>p {
            font-size: .8rem;
            display: inline-block;
            padding: 0 10px;
            transform: translateY(10px);
            background: white;
        }
        .dark .footer {
            border-color: #303030;
        }
        .dark .footer>p {
            background: #0b0f19;
        }
        .acknowledgments h4{
            margin: 1.25em 0 .25em 0;
            font-weight: bold;
            font-size: 115%;
        }
        #container-advanced-btns{
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
        }
        .animate-spin {
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            from {
                transform: rotate(0deg);
            }
            to {
                transform: rotate(360deg);
            }
        }
        #share-btn-container {
            display: flex; padding-left: 0.5rem !important; padding-right: 0.5rem !important; background-color: #000000; justify-content: center; align-items: center; border-radius: 9999px !important; width: 13rem;
            margin-top: 10px;
            margin-left: auto;
        }
        #share-btn {
            all: initial; color: #ffffff;font-weight: 600; cursor:pointer; font-family: 'IBM Plex Sans', sans-serif; margin-left: 0.5rem !important; padding-top: 0.25rem !important; padding-bottom: 0.25rem !important;right:0;
        }
        #share-btn * {
            all: unset;
        }
        #share-btn-container div:nth-child(-n+2){
            width: auto !important;
            min-height: 0px !important;
        }
        #share-btn-container .wrap {
            display: none !important;
        }
        .gr-form{
            flex: 1 1 50%; border-top-right-radius: 0; border-bottom-right-radius: 0;
        }
        #prompt-container{
            gap: 0;
        }
        #generated_id{
            min-height: 700px
        }
        #setting_id{
          margin-bottom: 12px;
          text-align: center;
          font-weight: 900;
        }
"""
iface = gr.Blocks(css=css)

with iface:
    gr.HTML(
        """
            <div style="text-align: center; max-width: 700px; margin: 0 auto;">
              <div
                style="
                  display: inline-flex;
                  align-items: center;
                  gap: 0.8rem;
                  font-size: 1.75rem;
                "
              >
                <h1 style="font-weight: 900; margin-bottom: 7px; line-height: normal;">
                  V2A-Mapper: Image-to-Audio Generation
                </h1>
              </div>
            </div>
        """
    )
    with gr.Group():
        with gr.Box():

            image = gr.Image(type="pil", label="Upload Image", elem_id="input-image")
            with gr.Accordion("Click to modify detailed configurations", open=False):
                seed = gr.Number(
                    value=42,
                    label="Change this value (any integer number) will lead to a different generation result.",
                )
                duration = gr.Slider(
                    2.5, 10, value=5, step=2.5, label="Duration (seconds)"
                )
                guidance_scale = gr.Slider(
                    0,
                    5,
                    value=2.5,
                    step=0.5,
                    label="Guidance scale (Large => better quality and relavancy to text; Small => better diversity)",
                )
                n_candidates = gr.Slider(
                    1,
                    5,
                    value=3,
                    step=1,
                    label="Automatic quality control. This number control the number of candidates (e.g., generate three audios and choose the best to show you). A Larger value usually lead to better quality with heavier computation",
                )
                model_name = gr.Dropdown(
                    ["audioldm-s-full", "audioldm-l-full", "audioldm-s-full-v2","audioldm-m-text-ft", "audioldm-s-text-ft", "audioldm-m-full"], value="audioldm-m-full", label="Choose the model to use. audioldm-m-text-ft and audioldm-s-text-ft are recommanded. -s- means small, -m- means medium and -l- means large",
                )

            outputs = gr.Video(label="Output", elem_id="output-video")        
            btn = gr.Button("Submit").style(full_width=True)

        btn.click(
            image2audio,
            inputs=[image, duration, guidance_scale, seed, n_candidates, model_name],
            outputs=[outputs],
        )

        # with gr.Accordion("Additional information", open=False):
        #     gr.HTML(
        #         """
        #         <div class="acknowledgments">
        #             <p> We build the model with data from <a href="http://research.google.com/audioset/">AudioSet</a>, <a href="https://freesound.org/">Freesound</a> and <a href="https://sound-effects.bbcrewind.co.uk/">BBC Sound Effect library</a>. We share this demo based on the <a href="https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/375954/Research.pdf">UK copyright exception</a> of data for academic research. </p>
        #                     </div>
        #                 """
        #     )
            
iface.queue(concurrency_count=3)
iface.launch(debug=True, share=False)
