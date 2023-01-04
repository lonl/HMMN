import numpy as np

class Sample(object):
    def __init__(self,pic ,mean ,label, ori_pic):
        self.pic = pic
        self.mean = mean
        self.label = label

        self.ori_pic = ori_pic

class MEMORY2(object):
    def __init__(self, nSample, feature_dim, filter):
        self.nSample = nSample

        self.distance_matrix = np.zeros((self.nSample, self.nSample)).astype(object)
        self.distance_matrix[:, :] = float("INF")

        self.gram_matrix = np.zeros((self.nSample, self.nSample)).astype(object)
        self.gram_matrix[:, :] = float("INF")

        self.samples_f = np.zeros((self.nSample, feature_dim, filter[0], filter[1]))


        self.prior_weights = np.zeros((self.nSample))

        self.num_training_samples = 0
        self.minmum_sample_weight = 0.0012
        self.learning_rate = 0.009

        self.merge_sample_id = -1
        self.new_sample_id = -1

        self.sample_store = {}


    def update_memory(self, new_train_sample, new_picture_sample, new_sample_label, new_ori_pic=None):
        self.merge_sample_id = -1
        self.new_sample_id = -1

        gram_vector = self.find_gram_vector(new_train_sample)

        new_train_sample_norm = ((new_train_sample * new_train_sample.conjugate()).sum())

        dist_vec = np.zeros((self.nSample, 1)).astype(object)

        # temp =  np.array(new_train_sample_norm) + np.diag(self.gram_matrix) - 2 * gram_vector[0, :]
        temp = (np.array(new_train_sample_norm) * np.diag(self.gram_matrix)) / np.square(gram_vector[0, :])
        temp[temp < 0] = 0

        dist_vec[:self.num_training_samples, :] = temp[:self.num_training_samples, np.newaxis]
        dist_vec[self.num_training_samples:, :] = float("INF")

        if self.num_training_samples == self.nSample:
            min_sample_weight, min_sample_id = self.findMin()

            # print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
            # print min_sample_weight, self.minmum_sample_weight
            # print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

            #if min_sample_weight < self.minmum_sample_weight:
            if False:

                self.update_distance_matrix(gram_vector, new_train_sample_norm, min_sample_id, -1, 0, 1)
                self.prior_weights[min_sample_id] = 0
                sum_ = np.sum(self.prior_weights)
                # print "sssssssssssssssssum:", sum_
                self.prior_weights = self.prior_weights * (1 - self.learning_rate) / sum_
                self.prior_weights[min_sample_id] = self.learning_rate

                # return min_sample_id, -1, 0, 1, "replace"

                # for st_id, s in enumerate(self.sample_store):
                #     if s.mean == min_sample_id:
                #         self.sample_store.pop(st_id)
                sam = Sample(new_picture_sample, min_sample_id, new_sample_label, new_ori_pic)

                #print("case 1: ")
                self.sample_store.pop(min_sample_id)
                l2 = []
                l2.append(sam)
                self.sample_store[min_sample_id] = l2

                #self.sample_store.append(sam)


            #else:
            if True:
                new_sample_min_dist = np.min(dist_vec)
                min_sample_id = np.argmin(dist_vec)

                duplicate = self.distance_matrix
                existing_samples_min_dist = np.min(duplicate)
                closest_exist_sample_pair = np.where(duplicate == np.min(duplicate))

                # print dist_vec
                # print existing_samples_min_dist
                # print closest_exist_sample_pair
                # print (dist_vec)
                #
                # print ("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
                # print (new_sample_min_dist, existing_samples_min_dist)
                # print ("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

                if new_sample_min_dist < existing_samples_min_dist:

                    # print min_sample_id
                    self.prior_weights = self.prior_weights * (1 - self.learning_rate)

                    merged_sample_id = min_sample_id

                    existing_sample_to_merge = self.samples_f[merged_sample_id]

                    merged_sample = self.merge_samples(existing_sample_to_merge, new_train_sample,
                                                       self.prior_weights[merged_sample_id], self.learning_rate,
                                                       "merge")
                    self.update_distance_matrix(gram_vector, new_train_sample_norm, merged_sample_id, -1,
                                                self.prior_weights[merged_sample_id], self.learning_rate)
                    self.prior_weights[merged_sample_id] += self.learning_rate

                    self.merge_sample_id = merged_sample_id

                    self.replace_sample(merged_sample, self.merge_sample_id)

                    # return merged_sample_id, -1, self.prior_weights[merged_sample_id], self.learning_rate, "merge"

                    sam = Sample(new_picture_sample, merged_sample_id, new_sample_label, new_ori_pic)
                    #self.sample_store.append(sam)
                    #print("case 2: ")
                    l1 = self.sample_store[merged_sample_id]
                    l1.append(sam)


                else:
                    self.prior_weights *= (1 - self.learning_rate)
                    if self.prior_weights[closest_exist_sample_pair[0][0]] > self.prior_weights[
                        closest_exist_sample_pair[0][1]]:

                        t = closest_exist_sample_pair[0][0]
                        closest_exist_sample_pair[0][0] = closest_exist_sample_pair[0][1]
                        closest_exist_sample_pair[0][1] = t

                    merged_sample = self.merge_samples(self.samples_f[closest_exist_sample_pair[0][0]],
                                                       self.samples_f[closest_exist_sample_pair[0][1]],
                                                       self.prior_weights[closest_exist_sample_pair[0][0]],
                                                       self.prior_weights[closest_exist_sample_pair[0][1]], "merge")

                    self.update_distance_matrix(gram_vector, new_train_sample_norm, closest_exist_sample_pair[0][0],
                                                closest_exist_sample_pair[0][1],
                                                self.prior_weights[closest_exist_sample_pair[0][0]],
                                                self.prior_weights[closest_exist_sample_pair[0][1]])

                    self.prior_weights[closest_exist_sample_pair[0][0]] += self.prior_weights[
                        closest_exist_sample_pair[0][1]]
                    self.prior_weights[closest_exist_sample_pair[0][1]] = self.learning_rate

                    self.merge_sample_id = closest_exist_sample_pair[0][0]
                    self.new_sample_id = closest_exist_sample_pair[0][1]

                    self.replace_sample(merged_sample, self.merge_sample_id)
                    self.replace_sample(new_train_sample, self.new_sample_id)

                    # return closest_exist_sample_pair[0][0], closest_exist_sample_pair[0][1], self.prior_weights[closest_exist_sample_pair[0][0]], self.prior_weights[closest_exist_sample_pair[0][1]], "merge"

                    # print self.samples_f

                    # for s in self.sample_store:
                    #     if s.mean == closest_exist_sample_pair[0][1]:
                    #         s.mean = closest_exist_sample_pair[0][0]
                    sam = Sample(new_picture_sample, closest_exist_sample_pair[0][1], new_sample_label, new_ori_pic)
                    #self.sample_store.append(sam)
                    #print("case 3: ")

                    l1 = self.sample_store[closest_exist_sample_pair[0][1]]
                    l2 = self.sample_store[closest_exist_sample_pair[0][0]]

                    if l1 == l2:
                        print ("error for memory")
                        print (self.gram_matrix)

                    for item in l1:
                        item.mean = closest_exist_sample_pair[0][0]
                        l2.append(item)

                    self.sample_store.pop(closest_exist_sample_pair[0][1])
                    l3 = []
                    l3.append(sam)
                    self.sample_store[closest_exist_sample_pair[0][1]] = l3

        else:
            sample_position = self.num_training_samples
            self.update_distance_matrix(gram_vector, new_train_sample_norm, sample_position, -1, 0, 1)

            if sample_position == 0:
                self.prior_weights[sample_position] = self.learning_rate

            else:
                self.prior_weights = self.prior_weights * (1 - self.learning_rate)
                self.prior_weights[sample_position] = self.learning_rate

            self.new_sample_id = sample_position

            self.replace_sample(new_train_sample, self.new_sample_id)

            sam = Sample(new_picture_sample, self.num_training_samples, new_sample_label, new_ori_pic)
            #self.sample_store.append(sam)

            l2 = []
            l2.append(sam)
            self.sample_store[self.num_training_samples] = l2

            self.num_training_samples += 1

    def find_gram_vector(self, new_train_sample):
        result = np.zeros((1, self.nSample)).astype(object)
        result[:, :] = float("inf")

        dis_vec = []
        for i in range(self.num_training_samples):
            dis_vec.append(self.feat_dis_compute(self.samples_f[i], new_train_sample))

        size = len(dis_vec)
        dis_vec = np.array(dis_vec)

        result[:, :size] = dis_vec
        return result

    def feat_dis_compute(self, feat1, feat2):
        if feat1.shape != feat2.shape:
            return 0

        dist = (feat1 * feat2).sum()  # fixme
        return dist

    def update_distance_matrix(self, gram_vector, new_sample_norm, id1, id2, w1, w2):
        alpha1 = w1 / (w1 + w2)
        alpha2 = 1 - alpha1

        if id2 < 0:

            norm_id1 = self.gram_matrix[id1, id1]
            if alpha1 == 0:
                self.gram_matrix[:, id1] = gram_vector[0]
                self.gram_matrix[id1, :] = gram_vector[0]
                self.gram_matrix[id1, id1] = new_sample_norm

            elif alpha2 == 0:
                pass
            else:

                t = alpha1 * self.gram_matrix[:, id1] + alpha2 * gram_vector[0]

                self.gram_matrix[:, id1] = np.squeeze(t)
                self.gram_matrix[id1, :] = np.squeeze(t)

                self.gram_matrix[
                    id1, id1] = alpha1 * alpha1 * norm_id1 + alpha2 * alpha2 * new_sample_norm + 2 * alpha1 * alpha2 * \
                                                                                                 gram_vector[
                                                                                                     0, id1]  # fixme

            dist_vec = np.zeros((self.nSample, 1))

            # temp = self.gram_matrix[id1, id1] + np.diag(self.gram_matrix) - 2 * self.gram_matrix[:, id1]
            temp = (self.gram_matrix[id1, id1] * np.diag(self.gram_matrix)) / np.square(self.gram_matrix[:, id1])

            # print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
            #
            # print alpha1, alpha2, gram_vector[0, id1], norm_id1, new_sample_norm
            #
            # print self.gram_matrix[:, id1]
            # print self.gram_matrix[id1, id1]
            # print np.diag(self.gram_matrix)

            # print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

            temp[temp < 0] = 0
            dist_vec[:, 0] = temp

            self.distance_matrix[:, id1] = dist_vec[:, 0]
            # tt = np.transpose(dist_vec)
            self.distance_matrix[id1, :] = dist_vec[:, 0]
            self.distance_matrix[id1, id1] = float("INF")
            # print self.distance_matrix

        else:
            norm_id1 = self.gram_matrix[id1, id1]
            norm_id2 = self.gram_matrix[id2, id2]
            id1_id2 = self.gram_matrix[id1, id2]

            # print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

            t = alpha1 * self.gram_matrix[:, id1] + alpha2 * self.gram_matrix[:, id2]
            self.gram_matrix[:, id1] = np.squeeze(t)
            self.gram_matrix[id1, :] = np.squeeze(t)
            self.gram_matrix[
                id1, id1] = alpha1 * alpha1 * norm_id1 + alpha2 * alpha2 * norm_id2 + 2 * alpha1 * alpha2 * id1_id2  # fixme

            # print self.gram_matrix[id1, id1]
            # print gram_vector[:, id1], gram_vector[:, id2]

            gram_vector[:, id1] = alpha1 * gram_vector[:, id1] + alpha2 * gram_vector[:, id2]

            t = np.squeeze(gram_vector)
            self.gram_matrix[:, id2] = t
            self.gram_matrix[id2, :] = t
            self.gram_matrix[id2, id2] = new_sample_norm

            for id in [id1, id2]:
                # temp = self.gram_matrix[id,id] + np.diag(self.gram_matrix) - 2 * self.gram_matrix[:, id]
                temp = (self.gram_matrix[id, id] * np.diag(self.gram_matrix)) / np.square(self.gram_matrix[:, id])
                temp[temp < 0] = 0
                self.distance_matrix[id, :] = np.squeeze(temp)
                self.distance_matrix[:, id] = np.squeeze(temp)
                self.distance_matrix[id, id] = float("INF")

                # print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

    def findMin(self):
        pos = np.argmin(self.prior_weights)
        min_w = np.min(self.prior_weights)
        return min_w, pos

    def merge_samples(self, sample1, sample2, w1, w2, sample_merge_type):

        alpha1 = w1 / (w1 + w2)
        alpha2 = 1 - alpha1

        # print sample1, sample2, alpha1, alpha2

        if sample_merge_type == "replace":
            return sample1
        elif sample_merge_type == "merge":
            merged_sample = alpha1 * sample1 + alpha2 * sample2
            return merged_sample

    def replace_sample(self, new_sample, idx):
        self.samples_f[idx] = new_sample

    def set_gram_matrix(self, r, c, val):
        self.gram_matrix[r, c] = val

    def get_merge_id(self):
        return self.merge_sample_id

    def get_new_id(self):
        return self.new_sample_id