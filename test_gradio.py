import os
import gradio as gr
import datetime
import base64
import requests
import json
import time
import wave
import io
from pathlib import Path
from typing import Optional, Dict, Tuple
import tempfile
import argparse # Moved from parse_args into the top-level imports


try:
    from inference_optimized import OptimizedInferenceAgent, InferenceOptions
    LOCAL_MODE_AVAILABLE = True
except ImportError:
    LOCAL_MODE_AVAILABLE = False
    print("⚠️ Local inference not available. API mode only.")

class GradioAPIInterface:
    def __init__(self, api_url: str = "https://ai.him-world.com", use_local: bool = False): # Fixed: init -> __init__
        self.api_url = api_url.rstrip('/')
        self.use_local = use_local and LOCAL_MODE_AVAILABLE

        if self.use_local: # Fixed: Indentation
            print("🔧 Initializing LOCAL inference mode...")
            self.opt = InferenceOptions().parse()
            self.opt.rank, self.opt.ngpus = 0, 1
            self.opt.optimization_mode = 'torch_compile'
            self.opt.use_fp16 = True
            os.makedirs(self.opt.res_dir, exist_ok=True)
            self.agent = OptimizedInferenceAgent(self.opt)
            print("✅ Local model loaded")
        else:
            print(f"🌐 Using API mode: {api_url}")  
            self.agent = None
        
        # Check API health
        if not self.use_local:
            self._check_api_health()

    def _check_api_health(self):
        """Check if API is available"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ API is healthy: {data.get('status')}")
                if 'optimization' in data:
                    print(f"⚡ Optimizations: {data['optimization']}")
            else:
                print(f"⚠️ API returned status {response.status_code}")
        except Exception as e:
            print(f"❌ Cannot reach API at {self.api_url}: {e}")

    def _encode_audio_to_base64(self, audio_path: str) -> Tuple[str, Dict]:
        """Encode audio file to base64 with metadata and validation"""
        try:
            # First, try to read as WAV
            try:
                with wave.open(audio_path, 'rb') as wav_file:
                    sample_rate = wav_file.getframerate()
                    channels = wav_file.getnchannels()
                    sample_width = wav_file.getsampwidth()
                    audio_data = wav_file.readframes(wav_file.getnframes())
                
                # Validate
                if len(audio_data) == 0:
                    raise ValueError("Empty audio data")
                
                # Encode to base64
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                
                metadata = {
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "sample_width": sample_width,
                    "format": "wav"
                }
                
                print(f"✅ Audio encoded: {len(audio_data)} bytes, {sample_rate}Hz, {channels}ch")
                return audio_base64, metadata
                
            except Exception as wav_error:
                print(f"⚠️ WAV read failed: {wav_error}, trying generic method...")
                
                # Fallback: Use pydub to convert to WAV first
                from pydub import AudioSegment
                
                # Detect format from extension
                ext = Path(audio_path).suffix.lower().lstrip('.')
                if not ext:
                    ext = 'wav'
                
                audio = AudioSegment.from_file(audio_path, format=ext)
                
                # Standardize format
                audio = audio.set_frame_rate(16000)
                audio = audio.set_channels(1)
                audio = audio.set_sample_width(2)
                
                # Export as WAV
                wav_buffer = io.BytesIO()
                audio.export(wav_buffer, format='wav')
                wav_buffer.seek(0)
                
                # Read back the WAV
                with wave.open(wav_buffer, 'rb') as wav_file:
                    sample_rate = wav_file.getframerate()
                    channels = wav_file.getnchannels()
                    sample_width = wav_file.getsampwidth()
                    audio_data = wav_file.readframes(wav_file.getnframes())
                
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                
                metadata = {
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "sample_width": sample_width,
                    "format": "wav"
                }
                
                print(f"✅ Audio converted and encoded: {len(audio_data)} bytes")
                return audio_base64, metadata
                
        except Exception as e:
            print(f"❌ Audio encoding error: {e}")
            # Last resort: read as raw bytes
            with open(audio_path, 'rb') as f:
                audio_data = f.read()
            
            if len(audio_data) == 0:
                raise ValueError("Audio file is empty")
            
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            return audio_base64, {
                "sample_rate": 16000, 
                "channels": 1, 
                "sample_width": 2,
                "format": "wav"
            }

    def _encode_image_to_base64(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def generate_via_api(
        self,
        ref_image,
        audio_file,
        emotion,
        a_cfg_scale,
        r_cfg_scale,
        e_cfg_scale,
        nfe,
        seed,
        no_crop,
        quality,
        use_webrtc,
        progress=gr.Progress()
    ):
        """Generate video via API"""
        try:
            progress(0, desc="Preparing request...")
            
            # Validate inputs
            if audio_file is None:
                return None, None, "❌ Please upload an audio file"
            
            # Image is optional - will use default if not provided
            if ref_image is None:
                progress(0.1, desc="No image provided, will use default...")
            
            # Encode audio
            progress(0.2, desc="Encoding audio...")
            audio_base64, audio_metadata = self._encode_audio_to_base64(audio_file)
            
            # Encode image (optional)
            image_base64 = None
            if ref_image:
                progress(0.3, desc="Encoding image...")
                image_base64 = self._encode_image_to_base64(ref_image)
            else:
                progress(0.3, desc="Using default image (img.jpg)...")
            
            # Prepare request
            payload = {
                "audio_base64": audio_base64,
                "audio_format": "wav",
                "sample_rate": audio_metadata["sample_rate"],
                "channels": audio_metadata["channels"],
                "sample_width": audio_metadata["sample_width"],
                "image_base64": image_base64,
                "image_format": "jpg",
                "emotion": emotion,
                "nfe": nfe,
                "seed": seed,
                "a_cfg_scale": a_cfg_scale,
                "r_cfg_scale": r_cfg_scale,
                "e_cfg_scale": e_cfg_scale,
                "no_crop": no_crop,
                "compress": True,
                "quality": quality
            }
            
            progress(0.4, desc="Sending to API...")
            
            # Send request
            start_time = time.time()
            response = requests.post(
                f"{self.api_url}/generate",
                json=payload,
                timeout=300  # 5 minutes timeout
            )
            
            if response.status_code != 200:
                return None, None, f"❌ API Error: {response.status_code} - {response.text}"
            
            result = response.json()
            request_time = time.time() - start_time
            
            progress(0.8, desc="Downloading video...")
            
            # Download video
            download_url = result.get("download_url")
            if not download_url:
                return None, None, "❌ No download URL in response"
            
            video_response = requests.get(download_url, timeout=60)
            if video_response.status_code != 200:
                return None, None, f"❌ Failed to download video: {video_response.status_code}"
            
            # Save video locally
            temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            with open(temp_video_path, 'wb') as f:
                f.write(video_response.content)
            
            progress(1.0, desc="Complete!")
            
            # Build status message
            perf = result.get("performance", {})
            status_msg = f"""✅ Video generated successfully via API!

  

📊 Performance Metrics:
├─ Request Time: {request_time:.2f}s
├─ Preprocessing: {perf.get('preprocess_time', 0):.2f}s
├─ Inference: {perf.get('inference_time', 0):.2f}s
├─ Postprocessing: {perf.get('postprocess_time', 0):.2f}s
└─ Total Time: {perf.get('total_time', 0):.2f}s

⚡ Optimization: {result.get('optimization', 'N/A')}
🚀 FPS: {perf.get('fps', 0):.2f}
📁 File Size: {result.get('file_size_mb', 0):.2f} MB
📐 Resolution: {result.get('resolution', 'N/A')}

🔗 Stream URL: {result.get('stream_url', 'N/A')}
"""

            # WebRTC embed HTML
            webrtc_html = None
            if use_webrtc:
                request_id = result.get('request_id')
                webrtc_html = self._generate_webrtc_player(request_id)
            
            return temp_video_path, webrtc_html, status_msg # Cleaned up indentation and structure
            
        except requests.exceptions.Timeout:
            return None, None, "❌ Request timeout. API might be overloaded."
        except Exception as e:
            return None, None, f"❌ Error: {str(e)}"

    def generate_via_local(
        self,
        ref_image,
        audio_file,
        emotion,
        a_cfg_scale,
        r_cfg_scale,
        e_cfg_scale,
        nfe,
        seed,
        no_crop,
        progress=gr.Progress()
    ):
        """Generate video using local model"""
        
        try:
            progress(0, desc="Preparing inputs...")
            
            # Only audio is required
            if audio_file is None:
                return None, "❌ Please upload an audio file"
            
            # Image is optional
            if ref_image is None:
                progress(0.1, desc="No image provided, will use default img.jpg...")
            
            # Generate output filename
            video_name = Path(ref_image).stem if ref_image else "default"
            audio_name = Path(audio_file).stem
            call_time = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            
            res_video_path = os.path.join(
                self.opt.res_dir,
                f"{call_time}-{video_name}-{audio_name}-local.mp4"
            )
            
            progress(0.3, desc="Running local inference...")
            
            # Use default image if not provided
            image_path = ref_image if ref_image else "img.jpg"
            if not os.path.exists(image_path):
                return None, f"❌ Default image not found at {image_path}"
            
            start_time = time.time()
            output_path = self.agent.run_inference(
                res_video_path=res_video_path,
                ref_path=image_path,
                audio_path=audio_file,
                a_cfg_scale=a_cfg_scale,
                r_cfg_scale=r_cfg_scale,
                e_cfg_scale=e_cfg_scale,
                emo=emotion,
                nfe=nfe,
                no_crop=no_crop,
                seed=seed,
                verbose=True
            )
            
            total_time = time.time() - start_time
            
            progress(1.0, desc="Complete!")
            
            image_used = "Custom" if ref_image else "Default (img.jpg)"
            status_msg = f"""✅ Video generated successfully (LOCAL)!

  

⏱️ Total Time: {total_time:.2f}s
📁 Saved to: {output_path}
⚡ Optimization: {self.opt.optimization_mode} + FP16
🖼️ Image Used: {image_used}
"""
    
            return output_path, status_msg # Cleaned up indentation and structure
            
        except Exception as e:
            return None, f"❌ Error: {str(e)}"

    def _generate_webrtc_player(self, stream_id: str) -> str:
        """Generate WebRTC player HTML"""
        return f"""
        <div style="width: 100%; max-width: 640px; margin: 0 auto;">
            <h3 style="text-align: center;">🎥 WebRTC Live Stream</h3>
            <video id="webrtc-video" autoplay playsinline controls 
                   style="width: 100%; border-radius: 8px; background: #000;">
            </video>
            <div style="text-align: center; margin-top: 10px;">
                <button id="start-webrtc" style="padding: 10px 20px; font-size: 16px; cursor: pointer;">
                    ▶️ Start WebRTC Stream
                </button>
                <button id="stop-webrtc" style="padding: 10px 20px; font-size: 16px; cursor: pointer; margin-left: 10px;">
                    ⏹️ Stop Stream
                </button>
            </div>
            <p style="text-align: center; margin-top: 10px; color: #666;">
                Stream ID: <code>{stream_id}</code>
            </p>
        </div>
        
        <script>
        let pc = null;
        const API_URL = '{self.api_url}';
        const STREAM_ID = '{stream_id}';
        
        async function startWebRTC() {{
            const video = document.getElementById('webrtc-video');
            
            // Create peer connection
            pc = new RTCPeerConnection({{
                iceServers: [
                    {{ urls: 'stun:stun.l.google.com:19302' }},
                    {{ urls: 'stun:stun1.l.google.com:19302' }}
                ]
            }});
            
            // Handle incoming tracks
            pc.ontrack = (event) => {{
                console.log('Received track:', event.track.kind);
                if (event.streams && event.streams[0]) {{
                    video.srcObject = event.streams[0];
                }}
            }};
            
            // Create offer
            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);
            
            // Send offer to server
            const response = await fetch(API_URL + '/offer', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{
                    sdp: pc.localDescription.sdp,
                    type: pc.localDescription.type,
                    stream_id: STREAM_ID
                }})
            }});
            
            const answer = await response.json();
            await pc.setRemoteDescription(new RTCSessionDescription(answer));
            
            console.log('WebRTC connection established');
        }}
        
        function stopWebRTC() {{
            if (pc) {{
                pc.close();
                pc = null;
            }}
            const video = document.getElementById('webrtc-video');
            if (video.srcObject) {{
                video.srcObject.getTracks().forEach(track => track.stop());
                video.srcObject = null;
            }}
            console.log('WebRTC connection closed');
        }}

        document.getElementById('start-webrtc').addEventListener('click', startWebRTC);
        document.getElementById('stop-webrtc').addEventListener('click', stopWebRTC);
        </script>
        """

    

def create_interface(api_url: str = "https://ai.him-world.com", use_local: bool = False):
    """Create Gradio interface with API support"""
    
    interface = GradioAPIInterface(api_url=api_url, use_local=use_local) # Fixed: Indentation
    mode_text = "LOCAL" if use_local else "API"

    custom_css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .mode-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        font-size: 0.875rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    """

    with gr.Blocks(css=custom_css, title="FLOAT - API Client") as demo:
        
        mode_badge = f"<span class='mode-badge'>⚡ {mode_text} MODE</span>"
        
        gr.Markdown(
            f"""
            # 🎭 FLOAT: Audio-Driven Talking Face Generation {mode_badge}
            
            Generate realistic talking face videos using {'optimized local inference' if use_local else 'the API backend'}.
            {'WebRTC streaming available for real-time playback!' if not use_local else ''}
            
            💡 **Tip**: Reference image is optional - if not provided, `img.jpg` will be used as default.
            """
        )
        
        with gr.Row():
            # Left column - Inputs
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Input Files")
                
                ref_image = gr.Image(
                    label="Reference Image (Optional - uses img.jpg if not provided)",
                    type="filepath",
                    sources=["upload"],
                    height=300
                )
                gr.Markdown("*Upload a face image or leave empty to use default*")
                
                audio_file = gr.Audio(
                    label="Audio File",
                    type="filepath",
                    sources=["upload"]
                )
                
                with gr.Accordion("⚙️ Generation Parameters", open=True):
                    emotion = gr.Dropdown(
                        choices=['S2E', 'angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
                        value='S2E',
                        label="Emotion Control"
                    )
                    
                    nfe = gr.Slider(1, 50, value=10, step=1, label="NFE")
                    a_cfg_scale = gr.Slider(0.0, 5.0, value=2.0, step=0.1, label="Audio CFG")
                    r_cfg_scale = gr.Slider(0.0, 3.0, value=1.0, step=0.1, label="Reference CFG")
                    e_cfg_scale = gr.Slider(0.0, 3.0, value=1.0, step=0.1, label="Emotion CFG")
                    seed = gr.Slider(0, 10000, value=25, step=1, label="Seed")
                    no_crop = gr.Checkbox(label="Skip Face Cropping", value=False)
                
                if not use_local:
                    with gr.Accordion("🎥 Streaming Options", open=True):
                        quality = gr.Radio(
                            choices=["fast", "balanced", "quality"],
                            value="balanced",
                            label="Video Quality"
                        )
                        use_webrtc = gr.Checkbox(
                            label="Enable WebRTC Streaming",
                            value=False,
                            info="Show WebRTC player for live streaming"
                        )
            
            # Right column - Output
            with gr.Column(scale=1):
                gr.Markdown("### 🎬 Output")
                
                generate_btn = gr.Button(
                    f"🚀 Generate Video ({mode_text})",
                    variant="primary",
                    size="lg"
                )
                
                status_output = gr.Textbox(
                    label="Status",
                    placeholder="Status messages will appear here...",
                    lines=10
                )
                
                video_output = gr.Video(
                    label="Generated Video",
                    height=400
                )
                
                if not use_local:
                    webrtc_output = gr.HTML(
                        label="WebRTC Player",
                        visible=True
                    )
        
        # API Info
        if not use_local:
            with gr.Accordion("ℹ️ API Information", open=False):
                gr.Markdown(
                    f"""
                    ### API Configuration
                    
                    - **API URL**: `{api_url}`
                    - **Mode**: Remote API
                    - **WebRTC**: Supported
                    - **Optimizations**: Server-side
                    
                    ### Features
                    
                    - ⚡ Optimized inference backend
                    - 🎥 WebRTC real-time streaming
                    - 📊 Performance metrics
                    - 🗜️ Video compression
                    - 📐 Auto-resize to 256x256
                    
                    ### How to Use
                    
                    1. Upload reference image and audio
                    2. Adjust parameters as needed
                    3. Enable WebRTC for live streaming (optional)
                    4. Click Generate
                    5. Download or stream the result
                    """
                )
        else:
            with gr.Accordion("ℹ️ Local Mode Information", open=False):
                gr.Markdown(
                    """
                    ### Local Configuration
                    
                    - **Mode**: Local Inference
                    - **Optimizations**: torch.compile + FP16
                    - **WebRTC**: Not available in local mode
                    
                    ### Benefits
                    
                    - ⚡ No network latency
                    - 🔒 Complete privacy
                    - 🚀 Direct GPU access
                    """
                )
        
        # Event handlers
        if use_local:
            generate_btn.click(
                fn=interface.generate_via_local,
                inputs=[
                    ref_image, audio_file, emotion,
                    a_cfg_scale, r_cfg_scale, e_cfg_scale,
                    nfe, seed, no_crop
                ],
                outputs=[video_output, status_output]
            )
        else:
            generate_btn.click(
                fn=interface.generate_via_api,
                inputs=[
                    ref_image, audio_file, emotion,
                    a_cfg_scale, r_cfg_scale, e_cfg_scale,
                    nfe, seed, no_crop, quality, use_webrtc
                ],
                outputs=[video_output, webrtc_output, status_output]
            )

    return demo

  

def parse_args():
    """Parse command line arguments"""
    
    parser = argparse.ArgumentParser(description='FLOAT Gradio Interface') # Fixed: Indentation

    parser.add_argument('--api_url', type=str, default='https://ai.him-world.com',
                        help='API URL (default: https://ai.him-world.com')
    parser.add_argument('--use_local', action='store_true',
                        help='Use local inference instead of API')
    parser.add_argument('--port', type=int, default=7860,
                        help='Gradio port (default: 7860)')
    parser.add_argument('--share', action='store_true',
                        help='Create public share link')
    parser.add_argument('--server_name', type=str, default='0.0.0.0',
                        help='Server name (default: 0.0.0.0)')

    return parser.parse_args()

  

if __name__ == "__main__": # Fixed: Missing underscores in __name__
    args = parse_args() # Fixed: Indentation

    print("\n" + "="*60)
    print("🚀 Launching FLOAT Gradio Interface")
    print("="*60)
    print(f"📍 Mode: {'LOCAL' if args.use_local else 'API'}")
    if not args.use_local:
        print(f"🌐 API URL: {args.api_url}")
    print(f"🔗 Gradio Port: {args.port}")
    print("="*60 + "\n")

    demo = create_interface(api_url=args.api_url, use_local=args.use_local)

    demo.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share
    )