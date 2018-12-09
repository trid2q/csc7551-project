#ifndef _FILTER_
#define _FILTER_

#include "Volume.h"

namespace edu {
namespace tri {
namespace cnn {

class Filter : public Volume<float> {
public:
    Filter(int w, int d): Volume<float>(w, d) {
    }
    virtual ~Filter() {
    }
};

}
}
}

#endif
