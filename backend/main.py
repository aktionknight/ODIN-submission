from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from odin_core import Odin

app = FastAPI()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

odin = Odin()

@app.get("/")
def root():
    return {"message": "ODIN backend running 🚀"}

@app.get("/trajectories")
def get_trajectories():
    t = np.linspace(0, 10, 100)
    trajectories = [np.column_stack([t, np.sin(t) + i]).tolist() for i in range(3)]
    chosen_idx, scores = odin.predict([np.array(traj) for traj in trajectories])
    return {
        "trajectories": trajectories,
        "chosen_index": chosen_idx,
        "scores": scores,
        "log": odin.log
    }
