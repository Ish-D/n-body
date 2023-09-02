#include <renderer.hpp>

int main() {
    std::vector<Point> points;

    std::default_random_engine gen;
    std::uniform_real_distribution<float> position(-2.0f, 2.0f);
    std::uniform_real_distribution<float> color(0.0f, 1.0f);
    std::uniform_real_distribution<float> size(0.0f, 100.0f);

    points.reserve(10000);
    for (int i = 0; i < 10000; i++)
        points.emplace_back(
            Point{.pos = Pos{position(gen), position(gen), position(gen)}, .size = size(gen), .color = Color{color(gen), color(gen), color(gen)}});

    nBody sim(points.size(), points.data());
    Renderer renderer(sim);

    renderer.init();
    renderer.mainLoop();
    return 0;
}