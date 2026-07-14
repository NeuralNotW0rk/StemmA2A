import os
import torch
from engine.model_adapters.stylegan_adapter import StyleGANAdapter
from diffracture.topology.grating import Grating
from diffracture.topology.bending import InvertElement
from diffracture.actant import Actant

def test_stylegan_bending_flow():
    print("=== Testing StyleGAN2 Model Adapter & Bending Integration ===")
    
    # 1. Initialize Adapter
    adapter = StyleGANAdapter()
    
    # 2. Register Model (resolutions default to 256)
    print("Registering model...")
    model_info = adapter.register_model(
        name="Test StyleGAN2",
        checkpoint_path="", # Empty path triggers random initialization fallback
        size=256,
        channel_multiplier=2
    )
    
    # 3. Load Model
    print("Loading model...")
    adapter.load_model(model_info)
    assert adapter.model is not None, "Model failed to load!"
    
    # 4. Base Image Generation (no bending)
    print("Generating base image...")
    base_artifact, base_img = adapter.generate(truncation=0.7, seed=42)
    print(f"Base artifact created: ID={base_artifact.id}, file={base_artifact.file.path}")
    assert os.path.exists(base_artifact.file.path), "Base image file not written to temp path!"
    
    # 5. Integrate with Diffracture: Apply bending via Grating Hook Injection
    print("Constructing Grating and Bending Element...")
    # Target conv1.conv (the 512-channel modulated convolution inside the first StyledConv block)
    invert_el = InvertElement(address="conv1.conv", indices=[0, 1, 2, 3])
    grating = Grating()
    grating.add_element(invert_el)
    
    print("Activating Grating intervention on the StyleGAN2 generator...")
    actant = Actant(adapter.model)
    actant.activate(grating, injection_strategy="hook")
    
    print("Generating bent image...")
    bent_artifact, bent_img = adapter.generate(truncation=0.7, seed=42)
    
    print("Deactivating Grating...")
    actant.deactivate()
    
    # 6. Verify differences
    print(f"Bent artifact created: ID={bent_artifact.id}, file={bent_artifact.file.path}")
    assert os.path.exists(bent_artifact.file.path), "Bent image file not written to temp path!"
    
    # Verify that the generated image tensor values are different
    # (since the hook altered conv1 activations)
    diff = torch.abs(base_img - bent_img).sum().item()
    print(f"Total pixel difference between base and bent image: {diff}")
    assert diff > 1e-4, "Bending intervention did not alter generator output!"
    
    # Clean up temp files
    try:
        os.remove(base_artifact.file.path)
        os.rmdir(os.path.dirname(base_artifact.file.path))
        os.remove(bent_artifact.file.path)
        os.rmdir(os.path.dirname(bent_artifact.file.path))
        print("Temporary files cleaned up.")
    except Exception as e:
        print(f"Non-critical cleanup warning: {e}")
        
    print("\n[OK] StyleGAN2 Adapter and Network Bending Integration Verified successfully!")

if __name__ == "__main__":
    test_stylegan_bending_flow()
