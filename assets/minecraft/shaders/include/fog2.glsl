#version 150


vec4 linear_fog(vec4 inColor, float vertexDistance, float fogStart, float fogEnd, vec4 fogColor) {
    if(fogStart > 1.0) { // just to look nicer
        fogStart /= 2;
    }
    if(vertexDistance <= fogStart) {
        return inColor;
    }
    float density = 0.75;

    vec4 fcolor = fogColor;
    float lum = luma4(fcolor.rgb);
    vec3 diff = fcolor.rgb - lum;
    fcolor.rgb = fcolor.rgb + diff * (-lum * 0.25 + 0.50);

    float fogValue = vertexDistance < fogEnd ? smoothstep(fogStart, fogEnd, vertexDistance) : 1.0;

    vec4 fog = vec4(mix(inColor.rgb, fcolor.rgb * 0.75, 1.0 - clamp(exp2(pow(density * fogValue, 2.0) * -1.442695), 0, 1)), inColor.a);
//    vec4 fog = vec4(mix(inColor.rgb, fcolor.rgb, fogValue * fcolor.a), inColor.a);

    return fog;
}

float linear_fog_fade(float vertexDistance, float fogStart, float fogEnd) {
    if(vertexDistance <= fogStart) {
        return 1.0;
    } else if(vertexDistance >= fogEnd) {
        return 0.0;
    }

    return smoothstep(fogEnd, fogStart, vertexDistance);
}
