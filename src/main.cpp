#include <renderer.hpp>

int main() {
    std::vector<Point> points;
    points.reserve(100);
    for (int i = 0; i < 100; i++) {
        points.emplace_back(Point{Pos{i / 50.0f - 1, static_cast<float>(i % 10) / 5 - 1, 0.0f}, 40, Color{i / 100.0f, 0.0f, 0.51f}});
    }

    nBody sim(points.size(), points.data());
    Renderer renderer(sim);

    renderer.init();
    renderer.mainLoop();
    return 0;
}