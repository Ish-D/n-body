#include <renderer.hpp>

int main() {
    std::vector<Point> points;

    points.emplace_back(Point{
        Pos{-0.5f, -0.5f, 0.0f},
        Color{  200,     0,    0},
        100.0f
    });

    points.emplace_back(Point{
        Pos{0.5f, -0.5f,   0.0f},
        Color{0.0f,  0.0f, 200.0f},
        40
    });

    points.emplace_back(Point{
        Pos{0.5f,   0.5f, 0.0f},
        Color{0.0f, 100.0f, 0.0f},
        20
    });

    points.emplace_back(Point{
        Pos{-0.5f, 0.5f,  0.0f},
        Color{50.0f, 0.0f, 50.0f},
        50
    });

    nBody sim(points.size(), points.data());
    Renderer renderer(sim);

    renderer.init();
    renderer.mainLoop();
    return 0;
}