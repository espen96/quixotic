int imod(int val, int modulo) {
    return val - val / modulo * modulo;
}

ivec2 positionToPixel(vec3 position, vec2 ScreenSize, out bool inside, int discardModulo) {
    inside = true;
    ivec2 iScreenSize = ivec2(ScreenSize) - ivec2(0, 1);
    ivec3 iPosition = ivec3(floor(position));
    int area = iScreenSize.x * iScreenSize.y / 2;
    ivec3 sides = ivec3(int(pow(float(area), 1.0 / 3.0)));

    iPosition += sides / 2;

    if (clamp(iPosition, ivec3(0), sides - 1) != iPosition) {
        inside = false;
        return ivec2(-1);
    }

    int index = iPosition.x + iPosition.z * sides.x + iPosition.y * sides.x * sides.z;
    ivec2 result = ivec2(
        imod(index, iScreenSize.x / 2) * 2,
        index / (iScreenSize.x / 2) + 1
    );
    result.x += imod(result.y, 2);

    return result;
}