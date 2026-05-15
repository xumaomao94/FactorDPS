import time
import numpy as np
from tqdm import tqdm


class PSDGenerator:
    def __init__(self, num_peaks_per_psd=3, K=64, basis_type='g', seed=None) -> None:
        """PSD generator

        Parameters
        ----------
        num_peaks_per_psd : int, optional
            Number of peaks in the psd, by default 3
        K : int, optional
            Frequency channels for the psd, by default 64
        basis_type : str, optional
            Type of basis to be used for psd generation, by default 'g'
        """
        self.seed = seed
        self.__setup__(num_peaks_per_psd=num_peaks_per_psd,
                       K=K,
                       basis_type=basis_type)

    def __setup__(self, num_peaks_per_psd, K, basis_type) -> None:
        self.num_peaks_per_psd = num_peaks_per_psd
        self.K = K
        self.basis_type = basis_type

    def generate_psd(self, num):
        seed = int(sum(100 * np.array(time.localtime())))
        if self.seed is not None:
            s = np.random.RandomState(self.seed)
        else:
            s = np.random.RandomState(seed)
        indK = np.arange(1, self.K + 1)

        if self.basis_type == 'g':
            def Sx(f0, sigma): return np.exp(-(indK - f0)
                                             ** 2 / (2 * sigma ** 2))
        elif self.basis_type == 's':
            def Sx(f0, a): return np.sinc((indK - f0) / a) ** 2 * \
                (np.abs((indK - f0) / a) <= 1)
        else:
            raise Exception("Invalid basis type set for generating psd!!!")

        Ctrue = np.empty((self.K, 0))
        num_peaks_per_psd = 3 # XULE: should not fix this here

        for rr in range(num):
            psd_peaks = self.generate_random_array(2, self.K-5, num_peaks_per_psd)
            # some form of amplitude of each peak
            am = 0.5 + 1.5 * s.rand(num_peaks_per_psd, 1)
            c = am[0] * Sx(psd_peaks[0], 2 + 2 * s.rand())
            for q in range(1, num_peaks_per_psd):
                # mix the psd resulting from each peak
                c += am[q] * Sx(psd_peaks[q], 2 + 2 * s.rand())

            Ctrue = np.hstack((Ctrue, c.reshape(-1, 1)))

        Ctrue = Ctrue / np.linalg.norm(Ctrue, axis=0)

        return Ctrue

    @staticmethod
    def generate_random_array(a, b, size):
        '''Equally divide the range a to b into size partitions and sample a peak point from each partition'''
        numbs = []
        steps = np.ceil((b-a)/size)
        l1 = a
        l2 = a+steps
        for i in range(size):
            numbs.append(np.random.randint(l1, l2))
            l1 = l1+steps
            l2 = l2+steps
        return np.array(numbs)


if __name__ == "__main__":
    psd_generator = PSDGenerator(num_peaks_per_psd=3,
                                 K = 64,
                                 basis_type='s')
    psds = psd_generator.generate_psd(num=100)
    print(psds.shape)
    
    