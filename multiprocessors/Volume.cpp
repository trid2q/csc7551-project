#include "Volume.h"

#include <iostream>
#include "Filter.h"

using namespace std;

namespace edu {
namespace tri {
namespace cnn {

ostream& operator<<(ostream& os, Volume<float>& src) {
    os << "Volume Src:" << endl;
    for (int i = 0; i < src.depth(); ++i) {
        os << "Depth " << i << ":" << endl << src[i] << endl;
    }
    return os;
}
    
}
}
}

