import numpy as np

# Toggle me
results = np.load('scene_results.npy', allow_pickle=True).item()
print(results)

# Toggle me 
# eval_file = "evaluations.npz" 

# data = np.load(eval_file, allow_pickle=True)

# for key in data.keys():
#     if isinstance(data[key], np.ndarray):
#         print(f"\n{key} shape: {data[key].shape}")
        
#         if data[key].size < 10 or key in ["timesteps"]:
#             print(f"{key} values: {data[key]}")
#         else:
#             if data[key].dtype.kind in 'iufc':
#                 print(f"{key} mean: {np.mean(data[key])}")
#                 print(f"{key} min: {np.min(data[key])}")
#                 print(f"{key} max: {np.max(data[key])}")
#             elif key == "successes" or key == "route_completion" or key == "total_cost":
#                 print(f"{key} mean: {np.mean(np.array(data[key]).astype(float))}")