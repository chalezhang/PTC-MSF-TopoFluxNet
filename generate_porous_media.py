import json
from pathlib import Path
import numpy as np
from geometry import generate
from mesh_writer import write_mesh


def write_solver_controls(case,c):
    """Foundation 13 modular incompressibleFluid SIMPLE: identical p/U residual controls."""
    case=Path(case)
    (case/'system').mkdir(parents=True,exist_ok=True)
    (case/'system/fvSolution').write_text(
      'FoamFile {version 2.0; format ascii; class dictionary; object fvSolution;}\n'
      'solvers {\n'
      'p {solver GAMG; tolerance 1e-10; relTol 0; smoother GaussSeidel; '
      'nPreSweeps 0; nPostSweeps 2; cacheAgglomeration true; '
      'nCellsInCoarsestLevel 10; agglomerator faceAreaPair;}\n'
      'U {solver smoothSolver; smoother symGaussSeidel; tolerance 1e-10; relTol 0;}\n}\n'
      f'SIMPLE {{nNonOrthogonalCorrectors 0; residualControl {{ p {c.CONVERGENCE_P:.12g}; U {c.CONVERGENCE_U:.12g}; }} }}\n'
      'relaxationFactors {fields {p 0.3;} equations {U 0.7;}}\n')
    (case/'system/controlDict').write_text(
      'FoamFile {version 2.0; format ascii; class dictionary; object controlDict;}\n'
      'application foamRun;\nsolver incompressibleFluid;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\n'
      f'endTime {c.SIMPLE_MAX_STEPS};\ndeltaT 1;\nwriteControl timeStep;\n'
      f'writeInterval {c.SIMPLE_MAX_STEPS};\nwriteFormat ascii;\nwritePrecision 12;\nrunTimeModifiable false;\n')


def build_case(case,seed,shape,domain,c,*,precomputed_geometry=None):
    # 所有算法尺寸由配置传入，不包含硬编码的 128 或 256。
    nx,ny=map(int,shape)
    g=precomputed_geometry
    if g is None:
        g=generate(nx,ny,seed,porosity=c.POROSITY,feature_size=c.FEATURE_SIZE,
            domain=domain,r_min=c.CIRCLE_R_MIN,r_max=c.CIRCLE_R_MAX,
            gap=c.CIRCLE_GAP,allow_overlap=c.CIRCLE_ALLOW_OVERLAP,
            overlap_ratio=c.CIRCLE_OVERLAP_RATIO,
            tol=c.POROSITY_TOLERANCE,soft_tol=c.POROSITY_SOFT_TOLERANCE,max_iter=c.CIRCLE_MAX_ITER,
            max_attempts=c.CIRCLE_MAX_ATTEMPTS,circle_phase=c.CIRCLE_PHASE,
            require_percolation=c.REQUIRE_PERCOLATION,
            max_cfd_removed_fraction=c.MAX_CFD_REMOVAL_FRACTION)
    if g.material_void.shape!=(ny,nx):
        raise ValueError('预生成材料尺寸与计算网格不一致')
    case=Path(case)
    if case.exists():raise RuntimeError(f'算例已存在，不允许覆盖：{case}')
    case.mkdir(parents=True)
    # 不强行打通自然闭塞材料。只有自然存在左右贯通孔时才输出 CFD 网格。
    if g.diagnostics['cfd_ready']:
        n,npoints=write_mesh(case,g.cfd_fluid,domain,c.THICKNESS,2)
        if n!=int(g.cfd_fluid.sum()):raise AssertionError('polyMesh 与 CFD 流体掩膜不一致')
    else:
        n,npoints=0,0
        print(f'[几何] seed={seed}: 没有自然贯通孔隙，仅生成材料图片与诊断；禁止运行 CFD',flush=True)
    (case/'0').mkdir();(case/'system').mkdir();(case/'constant').mkdir(exist_ok=True)
    (case/'0/p').write_text('FoamFile {version 2.0; format ascii; class volScalarField; object p;}\n'
      'dimensions [0 2 -2 0 0 0 0];\ninternalField uniform 0;\nboundaryField {\n'
      f'inlet {{type fixedValue; value uniform {c.PRESSURE_DROP};}}\n'
      'outlet {type fixedValue; value uniform 0;}\n'
      'walls {type zeroGradient;}\nfrontAndBack {type empty;}\n}\n')
    (case/'0/U').write_text('FoamFile {version 2.0; format ascii; class volVectorField; object U;}\n'
      'dimensions [0 1 -1 0 0 0 0];\ninternalField uniform (0 0 0);\nboundaryField {\n'
      'inlet {type zeroGradient;}\noutlet {type zeroGradient;}\n'
      'walls {type fixedValue; value uniform (0 0 0);}\nfrontAndBack {type empty;}\n}\n')
    (case/'constant/physicalProperties').write_text('FoamFile {version 2.0; format ascii; class dictionary; object physicalProperties;}\n'
      f'viscosityModel constant;\nnu [0 2 -1 0 0 0 0] {c.NU};\n')
    (case/'constant/momentumTransport').write_text('FoamFile {version 2.0; format ascii; class dictionary; object momentumTransport;}\n'
      'simulationType laminar;\n')
    (case/'system/fvSchemes').write_text('FoamFile {version 2.0; format ascii; class dictionary; object fvSchemes;}\n'
      'ddtSchemes {default steadyState;}\ngradSchemes {default Gauss linear;}\n'
      'divSchemes {default none; div(phi,U) Gauss upwind; '
      'div((nuEff*dev2(T(grad(U))))) Gauss linear;}\n'
      'laplacianSchemes {default Gauss linear corrected;}\n'
      'interpolationSchemes {default linear;}\nsnGradSchemes {default corrected;}\n')
    write_solver_controls(case,c)
    d=g.diagnostics
    meta={'openfoam_version':'13','dim':2,'grid':[nx,ny],'domain':list(domain),'thickness':c.THICKNESS,'seed':seed,
          'porosity_requested':c.POROSITY,'porosity_realized':d['total_porosity'],
          'total_porosity':d['total_porosity'],'connected_porosity':d['connected_porosity'],
          'isolated_porosity':d['isolated_porosity'],'isolated_share':d['isolated_fraction_of_pores'],
          'feature_size':c.FEATURE_SIZE,'pressure_drop':c.PRESSURE_DROP,'nu':c.NU,
          'iterations':c.SIMPLE_MAX_STEPS,'n_cells':n,'n_points':npoints,
          
          'circle_phase':c.CIRCLE_PHASE,
          'circle_r_min':c.CIRCLE_R_MIN,'circle_r_max':c.CIRCLE_R_MAX,
          'circle_gap':c.CIRCLE_GAP,'circle_allow_overlap':c.CIRCLE_ALLOW_OVERLAP,
          'circle_overlap_ratio':c.CIRCLE_OVERLAP_RATIO,
          'porosity_tolerance':c.POROSITY_TOLERANCE,'porosity_soft_tolerance':c.POROSITY_SOFT_TOLERANCE,
          'porosity_soft_accepted':d.get('porosity_soft_accepted',False),
          'cfd_ready':d['cfd_ready'],
          'cfd_excluded_corner_pixels':d['cfd_excluded_corner_pixels'],
          'geometry_diagnostics':d,
          'cfd_excludes_isolated_pores':True}
    (case/'geometry.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    np.save(case/'mask.npy',g.cfd_fluid.astype(np.uint8))
    np.save(case/'material_void.npy',g.material_void.astype(np.uint8))
    np.save(case/'isolated_void.npy',g.isolated_void.astype(np.uint8))
    return meta
