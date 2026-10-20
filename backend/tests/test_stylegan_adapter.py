import os
import sys
import torch
import unittest
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from engine.model_adapters.stylegan_adapter import StyleGANAdapter
from diffracture.topology.grating import Grating
from diffracture.topology.bending import InvertElement
from diffracture.actant import Actant


class TestStyleGANAdapter(unittest.TestCase):
    def test_stylegan_bending_flow(self):
        # 1. Initialize Adapter
        adapter = StyleGANAdapter()
        
        # 2. Register Model (resolutions default to 256)
        model_info = adapter.register_model(
            name="Test StyleGAN2",
            checkpoint_path="", # Empty path triggers random initialization fallback
            size=256,
            channel_multiplier=2
        )
        
        # 3. Load Model
        adapter.load_model(model_info)
        self.assertIsNotNone(adapter.model, "Model failed to load!")
        
        # 4. Base Image Generation (no bending)
        base_artifact, base_img = adapter.generate(truncation=0.7, seed=42)
        self.assertTrue(os.path.exists(base_artifact.file.path), "Base image file not written to temp path!")
        
        # 5. Integrate with Diffracture: Apply bending via Grating Hook Injection
        invert_el = InvertElement(address="conv1.conv", indices=[0, 1, 2, 3])
        grating = Grating()
        grating.add_element(invert_el)
        
        actant = Actant(adapter.model)
        actant.activate(grating, injection_strategy="hook")
        
        bent_artifact, bent_img = adapter.generate(truncation=0.7, seed=42)
        actant.deactivate()
        
        # 6. Verify differences
        self.assertTrue(os.path.exists(bent_artifact.file.path), "Bent image file not written to temp path!")
        
        diff = torch.abs(base_img - bent_img).sum().item()
        self.assertGreater(diff, 1e-4, "Bending intervention did not alter generator output!")
        
        # Clean up temp files
        try:
            os.remove(base_artifact.file.path)
            os.rmdir(os.path.dirname(base_artifact.file.path))
            os.remove(bent_artifact.file.path)
            os.rmdir(os.path.dirname(bent_artifact.file.path))
        except Exception:
            pass


    def test_register_model_checkpoints(self):
        import tempfile
        import pickle
        from engine.model_adapters.stylegan_adapter import _compute_stylegan_uid, Generator

        adapter = StyleGANAdapter()
        gen = Generator(size=64, style_dim=512, n_mlp=8, channel_multiplier=1)
        state_dict = gen.state_dict()

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Test standard PyTorch .pt / .pth checkpoint
            pt_path = os.path.join(tmpdir, "model.pt")
            torch.save({"g_ema": state_dict}, pt_path)

            uid_pt = _compute_stylegan_uid(pt_path, adapter.uid_generator)
            self.assertTrue(len(uid_pt) > 0)
            
            model_info_pt = adapter.register_model(
                name="PT Model",
                checkpoint_path=pt_path
            )
            self.assertEqual(model_info_pt.checkpoint.uid, uid_pt)
            self.assertEqual(model_info_pt.config["size"], 64)

            # 2. Test raw pickle (.pkl) checkpoint
            pkl_path = os.path.join(tmpdir, "model.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump({"g_ema": state_dict}, f)

            uid_pkl = _compute_stylegan_uid(pkl_path, adapter.uid_generator)
            self.assertEqual(uid_pkl, uid_pt)

            model_info_pkl = adapter.register_model(
                name="PKL Model",
                checkpoint_path=pkl_path
            )
            self.assertEqual(model_info_pkl.checkpoint.uid, uid_pkl)
            self.assertEqual(model_info_pkl.config["size"], 64)

            # 3. Load from PKL model
            adapter.load_model(model_info_pkl)
            self.assertIsNotNone(adapter.model)


if __name__ == "__main__":
    unittest.main()

