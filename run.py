from GeoGuesserSystem import *

if __name__ == '__main__':

    BR = system_loader()
    BR.prepare_dataloaders()
    BR.train(50)
    save_system(BR)