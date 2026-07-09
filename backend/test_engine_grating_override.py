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
        # Invert element initially targeting cluster 1
        cluster_map = [
            {"cluster_index": 1, "feature_index": 4},
            {"cluster_index": 1, "feature_index": 5},
            {"cluster_index": 2, "feature_index": 6}
        ]
        invert_el = InvertElement(address="conv1.conv", indices=[], cluster=1, cluster_map=cluster_map)
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
        
        # 5. Run generation with dynamic overrides (change cluster to 2)
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
                        "cluster": 2
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
        
        # Verify grating override parameters
        overrides = context["gratings"][0].get("overrides")
        assert overrides is not None and len(overrides) == 1, "Overrides list in context 'gratings' should have 1 element!"
        override_meta = overrides[0].get("metadata")
        assert "resolved_indices" not in override_meta, "resolved_indices should not be in override metadata!"
        
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
        
        # 3. Test update_batch_labels recursive comparison
        print("Testing update_batch_labels recursive labeling...")
        from param_graph.elements.collections.batch_element import Batch
        from param_graph.elements.artifacts.audio_element import Audio
        
        batch_id = "test_batch_id"
        m1_id = "member_1"
        m2_id = "member_2"
        
        m1 = Audio(
            id=m1_id,
            name="Member 1",
            file=Asset(path="somepath1.wav", uid=m1_id),
            sample_rate=16000,
            duration=1.0,
            context={
                "gratings": [
                    {
                        "id": "grating_1",
                        "strength": 1.0,
                        "overrides": [
                            {
                                "address": "conv1.conv",
                                "metadata": {
                                    "cluster": 1
                                }
                            }
                        ]
                    }
                ],
                "model_id": "some_model"
            }
        )
        m2 = Audio(
            id=m2_id,
            name="Member 2",
            file=Asset(path="somepath2.wav", uid=m2_id),
            sample_rate=16000,
            duration=1.0,
            context={
                "gratings": [
                    {
                        "id": "grating_1",
                        "strength": 1.0,
                        "overrides": [
                            {
                                "address": "conv1.conv",
                                "metadata": {
                                    "cluster": 2
                                }
                            }
                        ]
                    }
                ],
                "model_id": "some_model"
            }
        )
        
        batch_node = Batch(id=batch_id, member_ids=[m1_id, m2_id], member_type="audio")
        
        g.add_element(m1)
        g.add_element(m2)
        g.add_element(batch_node)
        
        # Trigger update_batch_labels
        from app import update_batch_labels
        update_batch_labels(batch_id)
        
        # Verify labels/aliases
        m1_alias = g.G.nodes[m1_id].get("alias")
        m2_alias = g.G.nodes[m2_id].get("alias")
        
        assert m1_alias == "conv1.conv.cluster: 1", f"Expected 'conv1.conv.cluster: 1', got '{m1_alias}'"
        assert m2_alias == "conv1.conv.cluster: 2", f"Expected 'conv1.conv.cluster: 2', got '{m2_alias}'"
        print("Recursive batch member labeling verified successfully!")
        
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
