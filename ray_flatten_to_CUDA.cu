#include "ray_flatten_to_CUDA.cuh"

__host__
void ray::flatten(const std::shared_ptr<Primitive>& object, std::vector<float>* out, std::vector<PrimitiveType>* outDesc) {
    //Recursive step
    if (!object->isOperator()) {
        //We reached an anchor
        outDesc->push_back(object->getType());

        float d[MAX_PARAMS];
        size_t s;
        object->getData(d, &s);
        out->insert(out->end(), d, d + object->getSize());
        return;
    }

    std::shared_ptr<BinaryOperator> bo = std::dynamic_pointer_cast<BinaryOperator>(object);
    //Check left and right
    flatten(bo->getP1(), out, outDesc);
    flatten(bo->getP2(), out, outDesc);

    //push back parent
    outDesc->push_back(object->getType());
    float d[MAX_PARAMS];
    size_t s;
    object->getData(d, &s);
    out->insert(out->end(), d, d + object->getSize());

}
