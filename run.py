from GeoGuesserSystem import *

if __name__ == '__main__':

    BR = system_loader()

    BR.train(100)
    save_system(BR)