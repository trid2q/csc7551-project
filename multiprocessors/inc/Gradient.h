#ifndef __GRADIENT__

#define __GRADIENT__

#include <iostream>
//#include "Layer.h"

namespace edu {
namespace tri {
namespace cnn {

class Gradient {
    friend ostream& operator<<(ostream& s, const Gradient& src);
    friend istream& operator>>(istream& s, const Gradient& src);

public:
    static const float LEARNING_RATE = 0.01f;
    static const float MOMENTUM = 0.6f;
    static const float WEIGHT_DECAY = 0.001f;

    Gradient(): old_(0), new_(0) {
    }
    virtual ~Gradient() {}
    float value() const {
	return new_;
    }
    void value(float val) {
	new_ = val;
    }
    float old() const {
	return old_;
    }
    void old(float val) {
	old_ = val;
    }
    void update() {
	old_ = new_ + old_ * MOMENTUM;
    }
    float adjustWeight(float curWeight, float fIn = 1.0) {
	float m = new_ + old_* MOMENTUM;
	float newWeight = curWeight - LEARNING_RATE*(m * fIn + WEIGHT_DECAY*curWeight);
	return newWeight;
    }

private:
    float new_;
    float old_;
};

}
}
}

#endif
