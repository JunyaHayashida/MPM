import torch
from MPM_Net import MPMNet
import numpy as np
import cv2
import os


class trackp():
    def __init__(self, mag_th, itp, sigma, maxv, image_size):
        self.MAG_TH = mag_th
        self.ATP_UTV = itp
        self.SIGMA = sigma
        self.MAXV = maxv
        self.IMAGE_SIZE = image_size

    def loadModel(self, model_path, parallel=True):
        '''load UNet-like model
        Args:
             model_path (str): model path (.pth)
             parallel (bool): for multi-gpu
        Return:
            model
        '''

        model = MPMNet(2, 3)
        model.cuda()
        state_dict = torch.load(model_path)
        if parallel:
            model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def inferenceMPM(self, names, model):
        '''
        Args:
             names (list of str): list of image path
             model (nn.Module): torch model
        Return:
            mpm (numpy.ndarray): mpm (HEIGHT, WIDTH, 3), ch0: y_flow, ch1: x_flow, ch2: z_flow
        '''

        imgs = [(cv2.imread(name, -1) / self.MAXV).astype('float32')[None] for name in names]
        img = np.concatenate(imgs, axis=0)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.cuda()
        output = model(img)
        mpm = output[0].cpu().detach().numpy()
        return np.transpose(mpm, axes=[1, 2, 0])

    def getMPMpoints(self, mpm, mag, mag_max=1.0):
        '''
        Args:
            mpm (numpy.ndarray): MPM
            mag (numpy.ndarray): Magnitude of MPM i.e. heatmap
            mag_max: Maximum value of heatmap
        Return:
            result (numpy.ndarray): Table of peak coordinates, warped coordinates, and peak value [x, y, wx, wy, pv]
        '''
        mag[mag > mag_max] = mag_max
        map_left_top, map_top, map_right_top = np.zeros(mag.shape), np.zeros(mag.shape), np.zeros(mag.shape)
        map_left, map_right = np.zeros(mag.shape), np.zeros(mag.shape)
        map_left_bottom, map_bottom, map_right_bottom = np.zeros(mag.shape), np.zeros(mag.shape), np.zeros(mag.shape)
        map_left_top[1:, 1:], map_top[:, 1:], map_right_top[:-1, 1:] = mag[:-1, :-1], mag[:, :-1], mag[1:, :-1]
        map_left[1:, :], map_right[:-1, :] = mag[:-1, :], mag[1:, :]
        map_left_bottom[1:, :-1], map_bottom[:, :-1], map_left_bottom[1:, :-1] = mag[:-1, 1:], mag[:, 1:], mag[1:, 1:]
        peaks_binary = np.logical_and.reduce((
            mag >= map_left_top, mag >= map_top, mag >= map_right_top,
            mag >= map_left, mag > self.MAG_TH, mag >= map_right,
            mag >= map_left_bottom, mag >= map_bottom, mag >= map_right_bottom,
        ))
        _, _, _, center = cv2.connectedComponentsWithStats((peaks_binary * 1).astype('uint8'))
        center = center[1:]
        result = []
        for center_cell in center.astype('int'):
            vec = mpm[center_cell[1], center_cell[0]]
            mag_value = mag[center_cell[1], center_cell[0]]
            # vec = vec / np.linalg.norm(vec)
            x = 10000 if vec[2] == 0 else int(5 * (vec[1] / vec[2]))  # Calculate the motion vector x
            y = 10000 if vec[2] == 0 else int(5 * (vec[0] / vec[2]))  # Calculate the motion vector y
            result.append([center_cell[0], center_cell[1], center_cell[0] + x, center_cell[1] + y, mag_value])
        return result

    def associateCells(self, frame, pos, pre_pos, pre_mag, new_id):
        '''Associate objects
         Args:
             frame (int): current frame
             pos (array-like): position table (from self.getMPMpoints)
             pre_pos (array-like) previous position table
             pre_mag (numpy.ndarray): previous heatmap
             new_id (int): new id for new cells
        Returns:
            ass_tab: record of associated cells
            not_ass_tab: record of not associated cells
            div_tab: record of mitosis cells
            app_tab: record of appeared cells
            new_id (int): new id for new cells
        '''
        
        ass_tab = []
        ass_flag = np.zeros(len(pos))
        fin_flag = np.ones(len(pre_pos))

        # association part -------------------------------------------------------
        # process each position of current frame ---------------------------------
        for i, focus_pos in enumerate(pos):
            # x, y is estimated point moved to the peak
            x, y, fmag_value, mag_value = self._movePeak(pre_mag, focus_pos[2], focus_pos[3])
            if mag_value > self.MAG_TH:
                # distance_list = [np.sqrt((pospos[2] - x) ** 2 + (pospos[3] - y) ** 2) for pospos in pre_pos]
                distance_list = np.linalg.norm(pre_pos[:, 2:4] - np.array([[x, y]]), axis=1).tolist()
                if min(distance_list) < 10:
                    add_ass_df = []
                    min_idx = np.argmin(distance_list)
                    add_ass_df.extend([frame, pre_pos[min_idx][1]] + focus_pos)
                    add_ass_df.extend([fmag_value, mag_value, min(distance_list), pre_pos[min_idx][10]])
                    ass_tab.append(add_ass_df)
                    fin_flag[min_idx] = 0
                    ass_flag[i] = 1
        print(f'# of associated cell: {len(ass_tab)}')

        # appearance part --------------------------------------------------------
        new_pos = np.delete(pos, np.where(ass_flag != 0)[0], axis=0).tolist()
        app_tab = []
        for focus_new_pos in new_pos:
            add_new_df = [frame, new_id] + focus_new_pos + [0, 0, 0, 0]
            ass_tab.append(add_new_df)
            app_tab.append(add_new_df)
            new_id += 1
        ass_tab = np.array(ass_tab)
        print(f'appeared cell: {len(app_tab)}')

        # division part ----------------------------------------------------------
        div_tab = np.empty((0, 11))
        ids, count = np.unique(ass_tab[:, 1], axis=0, return_counts=True)
        two_or_more_ids = ids[count > 1]
        for tm_id in two_or_more_ids:
            d_candidates = ass_tab[ass_tab[:, 1] == tm_id]
            d_candidates = d_candidates[np.argsort(d_candidates[:, 7])]
            # Change not daughters id & add not daughters to app tab -------------
            for d in d_candidates[:-2]:
                d[1] = new_id
                ass_tab[:, 1][(ass_tab[:, 2] == d[2]) & (ass_tab[:, 3] == d[3])] = new_id
                app_tab.append(d.tolist())
                new_id += 1
            # Change daughters id & register parents id -------------------------
            for d in d_candidates[-2:]:
                ass_tab[:, 1][(ass_tab[:, 2] == d[2]) & (ass_tab[:, 3] == d[3])] = new_id
                ass_tab[:, 10][(ass_tab[:, 2] == d[2]) & (ass_tab[:, 3] == d[3])] = tm_id
                new_id += 1
            div_tab = np.append(div_tab, ass_tab[ass_tab[:, 10] == tm_id], axis=0)
        print(f'division cell: {len(div_tab)}')

        not_ass_tab = np.delete(pre_pos, np.where(fin_flag == 0)[0], axis=0)
        print(f'not associated cell: {len(not_ass_tab)}')

        return ass_tab, not_ass_tab, div_tab, np.array(app_tab), new_id

    def _movePeak(self, mag, x, y):
        max_idx = -1
        try:
            first_mag_value = mag[int(y), int(x)]
            if first_mag_value < self.MAG_TH:
                return int(x), int(y), 0, 0
        except IndexError:
            return int(x), int(y), 0, 0
        while max_idx != 2:
            points = [[y + 1, x], [y - 1, x], [y, x], [y, x - 1], [y, x + 1]]
            mags = []
            for p in points:
                try:
                    mags.append(mag[int(p[0]), int(p[1])])
                except IndexError:
                    mags.append(0)  # out of frame
            max_idx = np.argmax(mags)
            y, x = points[max_idx]
            if mags[max_idx] == mags[2]:
                break
        x, y = self._adjastPosition(x, y)
        return x, y, first_mag_value, mags[max_idx]

    def _adjastPosition(self, x, y):
        y = 0 if y < 0 else y
        y = self.IMAGE_SIZE[0] - 1 if y > self.IMAGE_SIZE[0] - 1 else y
        x = 0 if x < 0 else x
        x = self.IMAZE_SIZE[1] - 1 if x > self.IMAGE_SIZE[1] - 1 else x
        return int(x), int(y)

    def _reformatID(self, ids, log):
        new_ids = np.arange(1, len(ids) + 1)
        new_log = np.empty((0, 11))
        for each_log in log.copy():
            eid = each_log[1]
            pid = each_log[10]
            each_log[1] = new_ids[np.where(ids == eid)[0][0]]
            if pid != 0:
                try:
                    each_log[10] = new_ids[np.where(ids == pid)[0][0]]
                except:
                    print(pid)
            new_log = np.append(new_log, [each_log], axis=0)
        return new_log

    def saveCTK(self, log, not_ass_log, div_log, save_dir):
        log = log[np.argsort(log[:, 0])]
        ids = np.unique(log[:, 1])

        new_log = self._reformatID(ids, log)
        # new_not_ass_log = self.reformatID(ids, not_ass_tab)
        # new_div_log = self.reformatID(ids, div_tab)
        # new_ins_log = self.reformatID(ids, ins_log)

        save_dir = os.path.join(save_dir, 'track_log')
        os.makedirs(save_dir, exist_ok=True)
        np.savetxt(os.path.join(save_dir, 'new_log.num_frame'), new_log)
        np.savetxt(os.path.join(save_dir, 'tab.num_frame'), log)
        np.savetxt(os.path.join(save_dir, 'end_trajectory.num_frame'), not_ass_log)
        np.savetxt(os.path.join(save_dir, 'mitosis_event.num_frame'), div_log)

        np.savetxt(os.path.join(save_dir, 'tracking.states'), new_log[:, :4], fmt='%d')
        np.savetxt(os.path.join(save_dir, 'end_trajectory.num_frame'), not_ass_log, fmt='%d')
        np.savetxt(os.path.join(save_dir, 'mitosis_event.num_frame'), div_log, fmt='%d')

        tree = np.empty((0, 2))
        parents = np.unique(new_log[:, 10])
        for nid in np.unique(new_log[:, 1]):
            par = new_log[:, 10][new_log[:, 1] == nid][0]
            tree_id = [nid, par]
            tree = np.append(tree, [tree_id], axis=0)
        np.savetxt(os.path.join(save_dir, 'tracking.tree'), tree, fmt='%d')

