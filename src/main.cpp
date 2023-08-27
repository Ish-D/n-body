#include <renderer.hpp>

int main() {
    std::vector<nBody::point> points;

    points.push_back(nBody::point{.x = -0.5f, .y = -0.5f, .z = 0.0f, .size = 100});
    points.push_back(nBody::point{.x = 0.5f, .y = -0.5f, .z = 0.0f, .size = 40});
    points.push_back(nBody::point{.x = 0.5f, .y = 0.5f, .z = 0.0f, .size = 20});
    points.push_back(nBody::point{.x = -0.5f, .y = 0.5f, .z = 0.0f, .size = 50});

    nBody sim(points.size(), points.data());
    Renderer renderer(sim);

    renderer.init();
    renderer.mainLoop();
    return 0;
}