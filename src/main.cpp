#include <renderer.hpp>

int main() {
    std::vector<Point> points;
    // points.reserve(1);

    for (int i = 0; i < 100; i++) {
        // points.emplace_back(Point{
        //     Pos{-0.5f, -0.5f, 0.0f},
        //     Color{},
        //     20
        // });
        points.emplace_back(Point{
            Pos{-0.5f, -0.5f,     0.0f},
            Color{   50,     0, i * 2.0f},
            50.0f
        });
    }

    nBody sim(points.size(), points.data());
    Renderer renderer(sim);

    renderer.init();
    renderer.mainLoop();
    return 0;
}