import os
from pathlib import Path

ROOT=Path(__file__).resolve().parent
FOAM13_BASHRC=os.getenv('POREAI_FOAM13_BASHRC','/opt/openfoam13/etc/bashrc')

NX=int(os.getenv('POREAI_NX','128'))
NY=int(os.getenv('POREAI_NY','128'))
CELL_SIZE=float(os.getenv('POREAI_CELL_SIZE','0.0003125'))
LX=float(os.getenv('POREAI_LX',str(NX*CELL_SIZE)))
LY=float(os.getenv('POREAI_LY',str(NY*CELL_SIZE)))
DOMAIN=(LX,LY)
THICKNESS=float(os.getenv('POREAI_THICKNESS','0.001'))
POROSITY=float(os.getenv('POREAI_POROSITY','0.52'))
CIRCLE_PHASE=os.getenv('POREAI_CIRCLE_PHASE','solid')
FEATURE_SIZE=float(os.getenv('POREAI_FEATURE_SIZE','0.001'))
CIRCLE_R_MIN=float(os.getenv('POREAI_R_MIN',str(.9375*FEATURE_SIZE/CELL_SIZE)))
CIRCLE_R_MAX=float(os.getenv('POREAI_R_MAX',str(3.125*FEATURE_SIZE/CELL_SIZE)))
CIRCLE_ALLOW_OVERLAP=os.getenv('POREAI_ALLOW_OVERLAP','1').lower() not in ('0','false','no')
CIRCLE_OVERLAP_RATIO=float(os.getenv('POREAI_OVERLAP_RATIO','0.35'))
CIRCLE_GAP=float(os.getenv('POREAI_GAP','0'))
POROSITY_TOLERANCE=float(os.getenv('POREAI_POROSITY_TOL','0.005'))
POROSITY_SOFT_TOLERANCE=float(os.getenv('POREAI_POROSITY_SOFT_TOL','0.015'))
CIRCLE_MAX_ITER=int(os.getenv('POREAI_CIRCLE_MAX_ITER','22'))
CIRCLE_MAX_ATTEMPTS=int(os.getenv('POREAI_CIRCLE_MAX_ATTEMPTS','12000'))
REQUIRE_PERCOLATION=False
FORCE_CFD_READY_DATASET=True
GENERATE_MAX_RETRIES_PER_CASE=int(os.getenv('POREAI_GENERATE_MAX_RETRIES','30'))
MAX_CFD_REMOVAL_FRACTION=float(os.getenv('POREAI_MAX_CFD_REMOVAL_FRACTION','0.10'))
GENERATE_SEED_RETRY_STRIDE=10000

# No test seed belongs to a train or validation split. All samples must have
# separately converged OpenFOAM p/U labels to be included in metrics.
TRAIN_GEOMETRIES=int(os.getenv('POREAI_TRAIN_GEOMETRIES','256'))
VAL_GEOMETRIES=int(os.getenv('POREAI_VAL_GEOMETRIES','32'))
TEST_GEOMETRIES=int(os.getenv('POREAI_TEST_GEOMETRIES','32'))
FINAL_TEST_GEOMETRIES=int(os.getenv('POREAI_FINAL_TEST_GEOMETRIES','32'))
THROAT_TRAIN_GEOMETRIES=int(os.getenv('POREAI_THROAT_TRAIN_GEOMETRIES','64'))
THROAT_VAL_GEOMETRIES=int(os.getenv('POREAI_THROAT_VAL_GEOMETRIES','16'))
THROAT_CANDIDATES=int(os.getenv('POREAI_THROAT_CANDIDATES','6'))
THROAT_MAX_RADIUS=float(os.getenv('POREAI_THROAT_MAX_RADIUS','3.5'))
SPLITS={
 'train':tuple(range(101,101+TRAIN_GEOMETRIES)),
 'val':tuple(range(2001,2001+VAL_GEOMETRIES)),
 'test':tuple(range(10001,10001+TEST_GEOMETRIES)),
 'throat_train':tuple(range(3101,3101+THROAT_TRAIN_GEOMETRIES)),
 'throat_val':tuple(range(5101,5101+THROAT_VAL_GEOMETRIES)),
 'final_test':tuple(range(12001,12001+FINAL_TEST_GEOMETRIES)),
}
DEFAULT_SPLITS=('train','throat_train','val','throat_val','test')

NU=float(os.getenv('POREAI_NU','1e-5'))
PRESSURE_DROP=float(os.getenv('POREAI_PRESSURE_DROP','1e-5'))
SIMPLE_MAX_STEPS=int(os.getenv('POREAI_MAX_SIMPLE_STEPS','800'))
CONVERGENCE_P=float(os.getenv('POREAI_CONVERGENCE_P','1e-6'))
CONVERGENCE_U=float(os.getenv('POREAI_CONVERGENCE_U','1e-6'))
MAX_CONTINUITY_ERROR=float(os.getenv('POREAI_MAX_CONTINUITY_ERROR','1e-7'))

CPU_BUDGET=int(os.getenv('POREAI_CPU_BUDGET','14'))
TORCH_THREADS=int(os.getenv('POREAI_THREADS','14'))
GENERATE_WORKERS=int(os.getenv('POREAI_GENERATE_WORKERS','3'))
COLLECT_WORKERS=int(os.getenv('POREAI_COLLECT_WORKERS','7'))
COLLECT_THREADS_PER_CASE=int(os.getenv('POREAI_COLLECT_THREADS_PER_CASE','2'))
EPOCHS=int(os.getenv('POREAI_EPOCHS','400'))
PATIENCE=int(os.getenv('POREAI_PATIENCE','50'))
BATCH_SIZE=int(os.getenv('POREAI_BATCH_SIZE','4'))
FIGURE_WORKERS=int(os.getenv('POREAI_FIGURE_WORKERS','3'))
PLOT_TRAIN_AUGMENTATION=os.getenv('POREAI_PLOT_TRAIN_AUGMENTATION','1')=='1'
FIGURE_CHUNK_SIZE=int(os.getenv('POREAI_FIGURE_CHUNK_SIZE','8'))
MODEL_CHANNELS=int(os.getenv('POREAI_MODEL_CHANNELS','32'))

EXPERIMENT_MODELS=('unet','topoflux')
PRIMARY_MODEL_NAME='topoflux'
UNET_CHANNELS=int(os.getenv('POREAI_UNET_CHANNELS','32'))
GRAPH_DEPTH=int(os.getenv('POREAI_GRAPH_DEPTH','6'))
LEARNING_RATE=float(os.getenv('POREAI_LR','2e-4'))
WEIGHT_DECAY=float(os.getenv('POREAI_WEIGHT_DECAY','1e-4'))
TRAIN_SEED=int(os.getenv('POREAI_TRAIN_SEED','2027'))
THROAT_SAMPLE_WEIGHT=float(os.getenv('POREAI_THROAT_SAMPLE_WEIGHT','2'))
LOSS_W_P=float(os.getenv('POREAI_LOSS_W_P','1'))
LOSS_W_U=float(os.getenv('POREAI_LOSS_W_U','4'))
LOSS_W_GRAD=float(os.getenv('POREAI_LOSS_W_GRAD','0.5'))
LOSS_W_PGRAD=float(os.getenv('POREAI_LOSS_W_PGRAD','0.6'))
LOSS_W_THROAT=float(os.getenv('POREAI_LOSS_W_THROAT','3'))
LOSS_W_FLUX=float(os.getenv('POREAI_LOSS_W_FLUX','0.2'))
LOSS_W_DIV=float(os.getenv('POREAI_LOSS_W_DIV','0.5'))
LOSS_W_DIRECTION=float(os.getenv('POREAI_LOSS_W_DIRECTION','0.2'))
RENDER_DPI=int(os.getenv('POREAI_RENDER_DPI','125'))
RENDER_COLLECT=False

CASE_ROOT=ROOT/'cases/study'
DATA_ROOT=ROOT/'data/steady'
MODEL_ROOT=ROOT/'models'
RESULT_ROOT=ROOT/'results'


def configured_model_names():return EXPERIMENT_MODELS

def model_architecture(name):
    return {'unet':'plain_unet','topoflux':'conditioned_multiscale_topoflux'}[name]

def model_width(name):return UNET_CHANNELS if name=='unet' else MODEL_CHANNELS

def pressure_scale():return abs(PRESSURE_DROP)
def velocity_scale():return max(abs(PRESSURE_DROP)*FEATURE_SIZE**2/(NU*LX),1e-15)

def physical_config():
    return {'grid':[NX,NY],'domain':[LX,LY],'thickness':THICKNESS,
            'nu':NU,'pressure_drop':PRESSURE_DROP,'feature_size':FEATURE_SIZE,
            'circle_r_min':CIRCLE_R_MIN,'circle_r_max':CIRCLE_R_MAX,
            'circle_gap':CIRCLE_GAP,'circle_allow_overlap':CIRCLE_ALLOW_OVERLAP,
            'circle_overlap_ratio':CIRCLE_OVERLAP_RATIO,'circle_phase':CIRCLE_PHASE,
            'porosity':POROSITY,'max_steps':SIMPLE_MAX_STEPS,
            'convergence_p':CONVERGENCE_P,'convergence_U':CONVERGENCE_U,
            'continuity_limit':MAX_CONTINUITY_ERROR,'throat_radius':THROAT_MAX_RADIUS}

def validate():
    if min(NX,NY)<32 or max(NX,NY)>512 or NX%16 or NY%16:
        raise ValueError('Grid must be 32..512 and divisible by 16')
    if min(LX,LY,THICKNESS,NU,FEATURE_SIZE,PRESSURE_DROP,CELL_SIZE)<=0:
        raise ValueError('Non-positive physical quantity')
    if CIRCLE_PHASE not in ('void','solid') or not 0<POROSITY<1:
        raise ValueError('Invalid circle/porosity settings')
    if not 0<CIRCLE_R_MIN<=CIRCLE_R_MAX<min(NX,NY)/2 or CIRCLE_GAP<0:
        raise ValueError('Invalid geometry radii/gap')
    if not 0<=CIRCLE_OVERLAP_RATIO<1 or not 0<MAX_CFD_REMOVAL_FRACTION<=1:
        raise ValueError('Invalid geometric constraints')
    if POROSITY_SOFT_TOLERANCE<POROSITY_TOLERANCE or POROSITY_TOLERANCE<=0:
        raise ValueError('Invalid porosity tolerance')
    if min(GENERATE_MAX_RETRIES_PER_CASE,CIRCLE_MAX_ITER,CIRCLE_MAX_ATTEMPTS,THROAT_CANDIDATES)<1:
        raise ValueError('Invalid geometry retry budget')
    if min(len(x) for x in SPLITS.values())<1 or sum(map(len,SPLITS.values()))!=len(set(sum((list(s) for s in SPLITS.values()),[]))):
        raise ValueError('Empty or overlapping splits')
    if not 0<THROAT_MAX_RADIUS<12:
        raise ValueError('Invalid throat radius')
    if min(CPU_BUDGET,TORCH_THREADS,COLLECT_WORKERS,COLLECT_THREADS_PER_CASE)<1 or TORCH_THREADS>CPU_BUDGET or COLLECT_WORKERS*COLLECT_THREADS_PER_CASE>CPU_BUDGET:
        raise ValueError('CPU thread budget exceeded')
    if min(EPOCHS,PATIENCE,BATCH_SIZE,MODEL_CHANNELS)<1 or MODEL_CHANNELS%8:
        raise ValueError('Invalid training size')
    if UNET_CHANNELS<8 or UNET_CHANNELS%8 or not 1<=GRAPH_DEPTH<=32:
        raise ValueError('Invalid model width or graph depth')
    if GENERATE_WORKERS<1 or GENERATE_WORKERS>CPU_BUDGET:
        raise ValueError('Invalid geometry generation parallelism')
    if FIGURE_WORKERS<1 or FIGURE_CHUNK_SIZE<1 or FIGURE_WORKERS>CPU_BUDGET:
        raise ValueError('Invalid figure workers/chunk size')
    if min(LOSS_W_P,LOSS_W_U,LOSS_W_GRAD,LOSS_W_PGRAD,LOSS_W_THROAT,LOSS_W_FLUX,LOSS_W_DIV,LOSS_W_DIRECTION,THROAT_SAMPLE_WEIGHT)<0:
        raise ValueError('Negative loss or sample weight')
    if min(SIMPLE_MAX_STEPS,CONVERGENCE_P,CONVERGENCE_U,MAX_CONTINUITY_ERROR)<=0:
        raise ValueError('Invalid convergence controls')
