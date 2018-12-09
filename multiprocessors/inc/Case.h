#ifndef _CASE_
#define _CASE_

#include "Volume.h"

namespace edu {
namespace tri {
namespace cnn {

class Case {
    public:
        Case(Volume<float>* image, Volume<float>* output): image_(image), output_(output) {
        }
        virtual ~Case() {
	    delete image_;
	    delete output_;
        }
	Volume<float>* image() const {
	    return image_;
	}
	Volume<float>* output() const {
	    return output_;
	}

    private:
	Volume<float>* image_;
	Volume<float>* output_;
};

}
}
}

using namespace edu::tri::cnn;

#endif
