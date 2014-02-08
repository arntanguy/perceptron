#include <stdexcept>

struct LayerNotLinkedException : public std::runtime_error
{
    int layerNumber = 0;
    explicit LayerNotLinkedException() : std::runtime_error("Error: The layer hasn't been linked to its output layer at this point.") {};
    explicit LayerNotLinkedException(const std::string& msg):std::runtime_error(msg){};
};

