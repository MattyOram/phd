import numpy as np

from helpers import (
    compute_power_law_coeffs, compute_img_metrics, compute_fe_grid_data, fe_mets_from_mesh, 
    avg_frames_from_paths, compute_CA_overlap, compute_error, compute_cop_error, compute_test_fe_error, 
    apply_fit, compute_polynomial_coeffs,
)

def check_max_raw(frames, name):
    if round(np.max(frames)) == 255:
        print(f'Fully saturated {name} sensel !')

class Errors():
    def __init__(
            self,
            
            fe_data, 
            common_Fs, 
            cal_F_range,
            cal_1_Fs,
            cal_1_paths,
            test_1_Fs,
            test_1_paths,
            sensor_id, 
            raw_mask, 
            custom_masks, 
            hold_time, 
            avg_window, 
            cal_start_time, 
            test_start_time, 
            guide_wall_z, 
            sensor_offset, 
            bone,
            bone_vol,
            fit_type=('power_law', True), # (power_law/poly, weight_by_patch_size, poly_degree, poly_zero_intercept)

            print_raw_warning=True,
            use_F2=False,
            sensor_offset_y=0
        ):

        self.cal_frames = None
        self.cal_frames_mpa = None
        self.cal_mets = None

        self.test_frames = None
        self.test_frames_mpa = None
        self.test_mets = None

        self.fe_grid_data = None
        self.fe_grid_mets = None
        self.fe_mets = None

        self.errors = None


        #•••••••••••••••• CALIBRATION ••••••••••••••••#

        cal_Fs = sorted([x for x in cal_1_Fs if x>=cal_F_range[0] and x<=cal_F_range[1]])
        cal_frames = avg_frames_from_paths(cal_1_paths, cal_Fs, sensor_id, hold_time, cal_start_time, avg_window, custom_masks, raw_mask, use_F2)
        if print_raw_warning:
            check_max_raw(cal_frames[max(cal_Fs)], 'calibration')
        if fit_type[0] == 'power_law':
            coeffs = compute_power_law_coeffs(cal_frames, weight_by_patch_size=fit_type[1])
        elif fit_type[0] == 'poly':
            coeffs = compute_polynomial_coeffs(
                    cal_frames, degree=fit_type[2], zero_intercept=fit_type[3], weight_by_patch_size=fit_type[1]
                )
        else: raise


        cal_frames_mpa = apply_fit(coeffs, fit_type[0], cal_frames)
        cal_mets = {f: compute_img_metrics(frame) for f, frame in cal_frames_mpa.items()}
        cal_pred_Fs = {f: cal_mets[f]['F_pred'] for f in cal_mets}

        #compute_error(cal_pred_Fs)

        #•••••••••••••••• CALIBRATION ••••••••••••••••#



        #•••••••••••••••• TESTING ••••••••••••••••#

        test_Fs = sorted(test_1_Fs)
        test_frames = avg_frames_from_paths(test_1_paths, test_Fs, sensor_id, hold_time, test_start_time, avg_window, custom_masks, raw_mask, use_F2)
        if print_raw_warning:
            check_max_raw(test_frames[max(test_Fs)], 'test')
        test_frames_mpa = apply_fit(coeffs, fit_type[0], test_frames)
        test_mets = {f: compute_img_metrics(frame) for f, frame in test_frames_mpa.items()}
        test_pred_Fs = {f: test_mets[f]['F_pred'] for f in test_mets}
        #compute_error(test_pred_Fs)

        #•••••••••••••••• TESTING ••••••••••••••••#



        #•••••••••••••••• FE ••••••••••••••••#

        #fe_data = build_fe_data(inp_file)
        #common_Fs = set(fe_data) & set(test_frames)

        fe_grid_data = compute_fe_grid_data(common_Fs, fe_data, test_frames_mpa, bone, guide_wall_z, sensor_offset, sensor_offset_y)
        fe_grid_mets = {f: compute_img_metrics(frame) for f, frame in fe_grid_data.items()}
        fe_mets = {f: fe_mets_from_mesh(fe_data[f][bone]) for f in fe_data}

        #•••••••••••••••• FE ••••••••••••••••#



        #•••••••••••••••• RESULTS •••••••••••••••#

        # Calibration vs Instron force
        cal_error = compute_error(list(cal_pred_Fs.keys()), list(cal_pred_Fs.values()))

        # Testing vs Instron force
        test_error = compute_error(list(test_pred_Fs.keys()), list(test_pred_Fs.values()))

        # Testing vs FE
        metrics = ['P_max', 'P_avg', 'CA']#, 'COP']
        fe_error = compute_test_fe_error(metrics, common_Fs, test_mets, fe_mets)
        fe_grid_error = compute_test_fe_error(metrics, common_Fs, test_mets, fe_grid_mets)

            # COP error - only works for grid rn cos fe_COP is in simulation coords not grid idx/coords
        #fe_cop_error = compute_cop_error(test_mets, fe_mets, common_Fs, bone_vol, mode='raw')
        fe_grid_cop_error = compute_cop_error(test_mets, fe_grid_mets, common_Fs, bone_vol)

            # CA overlap
        ca_overlap = {f: compute_CA_overlap(fe_grid_data[f], test_frames_mpa[f]) for f in common_Fs}
        CAO_mean = np.mean(list(ca_overlap.values()))

        errors = {
            'cal_error':cal_error,
            'test_error': test_error, 
            'fe_error': fe_error, 
            'fe_grid_error': fe_grid_error, 
            'fe_grid_cop_error': fe_grid_cop_error, 
            'CAO_mean_pct': CAO_mean, 
        }

        self.cal_frames = cal_frames
        self.cal_frames_mpa = cal_frames_mpa
        self.cal_mets = cal_mets

        self.test_frames = test_frames
        self.test_frames_mpa = test_frames_mpa
        self.test_mets = test_mets

        self.fe_grid_data = fe_grid_data
        self.fe_grid_mets = fe_grid_mets
        self.fe_mets = fe_mets

        self.errors = errors

    #•••••••••••••••• RESULTS •••••••••••••••#