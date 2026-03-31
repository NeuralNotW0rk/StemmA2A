from collections import OrderedDict
from param_graph.elements.models.base_model_element import Model

# This is a bit of a hack. We're assuming the adapter is the state.
# A better approach would be to have the adapter be stateless and the
# model object itself be the state.
class ModelCache:
    def __init__(self, capacity=3):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, model: Model, adapter_class):
        model_id = model.id
        if model_id in self.cache:
            # Move to end to show it was recently used
            self.cache.move_to_end(model_id)
            return self.cache[model_id]
        
        # Not in cache, load it
        if len(self.cache) >= self.capacity:
            # Evict oldest item
            oldest_id, oldest_adapter = self.cache.popitem(last=False)
            print(f"Evicting model {oldest_id} from cache.")
            # In the future, we might want to call a cleanup method on the adapter
            # to release GPU memory explicitly.
            del oldest_adapter 

        print(f"Loading and caching model {model_id}.")
        adapter = adapter_class()
        adapter.load_model(model)
        self.cache[model_id] = adapter
        return adapter

