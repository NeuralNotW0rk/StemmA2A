import os
import asyncio
import tempfile
from pathlib import Path
import torch
import shutil

from engine.local_engine import LocalEngine
from param_graph.elements.models.stylegan_element import StyleGANModel
from param_graph.elements.artifacts.grating_element import Grating as GratingArtifact
from param_graph.elements.base_elements import Asset
from diffracture.topology.grating import Grating
from diffracture.topology.bending import InvertElement

from app import app, param_graph as app_param_graph, engine_provider as app_engine_provider
from param_graph.graph import ParameterGraph

async def test_grating_override():
    print("=== Testing Grating Dynamic Overrides inside LocalEngine ===")
    
    # 1. Initialize LocalEngine with temporary data root
    tmp_dir = tempfile.mkdtemp()
    engine = LocalEngine(data_root=tmp_dir)
    
    try:
        # 2. Register stylegan2 model
        print("Registering model...")
        model_info = await engine.register_model(
            "stylegan2",
            name="Test StyleGAN2",
            checkpoint_path="", # Triggers random initialization
            size=256,
            channel_multiplier=2
        )
        
        # 3. Create a temporary Grating checkpoint file with InvertElement
        print("Creating temporary Grating checkpoint file...")
        grating = Grating()
        # Invert element initially targeting indices [0, 1, 2, 3]
        invert_el = InvertElement(address="conv1.conv", indices=[0, 1, 2, 3])
        grating.add_element(invert_el)
        
        grating_file = os.path.join(tmp_dir, "test_grating.safetensors")
        grating.save(grating_file)
        
        # 4. Construct GratingArtifact
        grating_artifact = GratingArtifact(
            id="test-grating-id",
            name="Test Grating",
            context={},
            file=Asset(path=grating_file, uid="test-grating-uid", extension=".safetensors"),
            base_model_id=model_info.id,
            elements=[{
                "address": "conv1.conv",
                "kernel_type": "invert",
                "metadata": invert_el.metadata
            }]
        )
        
        # 5. Run generation with dynamic overrides (change indices to [4, 5, 6, 7])
        print("Submitting generate job with dynamic overrides...")
        job_id = await engine.execute(
            "generate",
            model_element=model_info,
            grating_elements=[grating_artifact],
            grating_strengths=[1.0],
            gratings=[{
                "id": grating_artifact.id,
                "strength": 1.0,
                "overrides": [{
                    "address": "conv1.conv",
                    "metadata": {
                        "indices": [4, 5, 6, 7]
                    }
                }]
            }],
            seed=42,
            truncation=0.7
        )
        
        # 6. Poll for job completion
        print("Waiting for job to complete...")
        while True:
            status = await engine.get_job_status(job_id)
            if status["status"] in ["completed", "failed"]:
                break
            await asyncio.sleep(0.5)
            
        print(f"Job finished with status: {status['status']}")
        if status["status"] == "failed":
            print(f"Error: {status.get('error')}")
            assert False, "Job failed!"
            
        result_artifact = status["result"]
        print(f"Output generated successfully! ID={result_artifact['id']}, Path={result_artifact['file']['path']}")
        
        # Verify context contains the grating parameters
        assert "context" in result_artifact, "Result artifact is missing 'context'!"
        context = result_artifact["context"]
        assert "gratings" in context, "Context is missing 'gratings' parameters!"
        assert len(context["gratings"]) == 1, "Context 'gratings' length is not 1!"
        assert context["gratings"][0]["id"] == grating_artifact.id, "Context grating ID mismatch!"
        assert "grating_ids" in context, "Context is missing 'grating_ids'!"
        assert context["grating_ids"] == [grating_artifact.id], "Context 'grating_ids' mismatch!"
        print("Grating parameters successfully verified in generated artifact context!")
        
    finally:
        # Clean up temp folder
        try:
            shutil.rmtree(tmp_dir)
            print("Temporary directory cleared.")
        except Exception as e:
            print(f"Non-critical cleanup warning: {e}")

def test_flask_endpoints():
    print("\n=== Testing Flask Model Layers & Grating Creation Endpoints ===")
    
    # Setup app context and mock param_graph
    tmp_dir = tempfile.mkdtemp()
    
    # Instantiate clean graph and engine provider
    from engine.engine_provider import EngineProvider
    import app as app_module
    
    EngineProvider._engine_instance = None
    g = ParameterGraph(tmp_dir)
    provider = EngineProvider(data_root=tmp_dir)
    
    old_graph = app_module.param_graph
    old_provider = app_module.engine_provider
    app_module.param_graph = g
    app_module.engine_provider = provider
    
    # Import and register model in the graph using asyncio.run
    engine = provider.get_engine()
    model_info = asyncio.run(engine.register_model(
        "stylegan2",
        name="Test StyleGAN2",
        checkpoint_path="",
        size=256,
        channel_multiplier=2
    ))
    g.add_element(model_info)
    g.save()
    
    client = app.test_client()
    
    try:
        # 1. Test GET /model/<model_id>/layers
        print("Testing GET /model/<model_id>/layers...")
        resp = client.get(f"/model/{model_info.id}/layers")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        data = resp.get_json()
        assert data["success"] is True
        assert len(data["layers"]) > 0, "No layers returned!"
        print(f"Returned {len(data['layers'])} layers successfully.")
        
        # Verify layer dict fields
        first_layer = data["layers"][0]
        assert "address" in first_layer
        assert "type" in first_layer
        
        # 2. Test POST /create_grating with clustering
        print("Testing POST /create_grating with feature clustering...")
        payload = {
            "model_id": model_info.id,
            "name": "Bended Erode Layer",
            "elements": [
                {
                    "address": "conv1.conv",
                    "kernel_type": "erode",
                    "params": {"radius": 2},
                    "perform_clustering": True,
                    "num_clusters": 3,
                    "cluster": 1
                }
            ]
        }
        resp = client.post("/create_grating", json=payload)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        data = resp.get_json()
        assert data["success"] is True
        assert "grating" in data
        
        grating_node = data["grating"]
        assert grating_node["id"].startswith("grating_")
        assert grating_node["base_model_id"] == model_info.id
        
        # Verify the created elements have a resolved cluster_map
        elements = grating_node["elements"]
        assert len(elements) == 1
        el = elements[0]
        assert el["address"] == "conv1.conv"
        assert el["kernel_type"] == "erode"
        assert el["metadata"]["cluster"] == 1
        assert el["metadata"]["cluster_map"] is not None
        assert len(el["metadata"]["cluster_map"]) > 0, "Cluster map was not populated!"
        print("Feature clustering and dynamic grating creation verified successfully!")
        
    finally:
        app_module.param_graph = old_graph
        app_module.engine_provider = old_provider
        try:
            shutil.rmtree(tmp_dir)
            print("Temporary test directory cleared.")
        except Exception as e:
            print(f"Non-critical cleanup warning: {e}")

if __name__ == "__main__":
    asyncio.run(test_grating_override())
    test_flask_endpoints()
