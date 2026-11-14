# OSDB_Data
OpenSeizureDatabase Data (to accompany the [OpenSeizureDatabase](http://github.com/OpenSeizureDetector/OpenSeizureDatabase) repository)

The OpenSeizureDatabase data is licenced under a variation of the Creative Commons Share Alike licence, with an additional licence condition to publish results obtained using the data - see the [LICENCE](https://github.com/OpenSeizureDetector/OpenSeizureDatabase/blob/main/documentation/LICENCE.md) description for more details.


NOTE:  A few of the seizures in the database are shown as using the 'Phone' data source.  This data should be treated with caution as we do not know where on the body the phone was located, and the 3d data seems to be incorrect (although the vector magnitude data is correct).

# Getting Started

The best thing to do is download the latest release of the database from here:  https://github.com/OpenSeizureDetector/OSDB_Data/releases/latest

(The reason you use the release rather than cloning the repository is that the false alarm dataset is too big to be in a git repository so I keep that separate and attach it to the releases - the git repository only contains the seizure data).

There is another repository https://github.com/OpenSeizureDetector/OpenSeizureDatabase which is a set of python scripts that I use to manipulate the data - the [README file](https://github.com/OpenSeizureDetector/OpenSeizureDatabase/blob/main/README.md) provides some instructions to get started using the database data and toolkit.   

We would be very pleased to receive pull requests for improvements to the tookit :).

Graham Jones (graham@openseizuredetector.org.uk)


