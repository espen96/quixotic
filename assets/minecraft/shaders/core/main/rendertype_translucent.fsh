#version 150

#moj_import <fog.glsl>
#moj_import <utils.glsl>

uniform sampler2D Sampler0;

uniform vec4 ColorModulator;
uniform float FogStart;
uniform float FogEnd;
uniform vec4 FogColor;

in float vertexDistance;
in float water;
in vec4 vertexColor;
in vec3 noise;
in vec3 color2;
in vec2 texCoord0;
in vec4 normal;
in vec4 glpos;

out vec4 fragColor;

float decodeFloat7_4(uint raw) {
    uint sign = raw >> 11u;
    uint exponent = (raw >> 7u) & 15u;
    uint mantissa = 128u | (raw & 127u);
    return (float(sign) * -2.0 + 1.0) * float(mantissa) * exp2(float(exponent) - 14.0);
}

float decodeFloat6_4(uint raw) {
    uint sign = raw >> 10u;
    uint exponent = (raw >> 6u) & 15u;
    uint mantissa = 64u | (raw & 63u);
    return (float(sign) * -2.0 + 1.0) * float(mantissa) * exp2(float(exponent) - 13.0);
}

vec3 decodeColor(vec4 raw) {
    uvec4 scaled = uvec4(round(raw * 255.0));
    uint encoded = (scaled.r << 24) | (scaled.g << 16) | (scaled.b << 8) | scaled.a;
    
    return vec3(
        decodeFloat7_4(encoded >> 21),
        decodeFloat7_4((encoded >> 10) & 2047u),
        decodeFloat6_4(encoded & 1023u)
    );
}

uint encodeFloat7_4(float val) {
    uint sign = val >= 0.0 ? 0u : 1u;
    uint exponent = uint(clamp(log2(abs(val)) + 7.0, 0.0, 15.0));
    uint mantissa = uint(abs(val) * exp2(-float(exponent) + 14.0)) & 127u;
    return (sign << 11u) | (exponent << 7u) | mantissa;
}

uint encodeFloat6_4(float val) {
    uint sign = val >= 0.0 ? 0u : 1u;
    uint exponent = uint(clamp(log2(abs(val)) + 7.0, 0.0, 15.0));
    uint mantissa = uint(abs(val) * exp2(-float(exponent) + 13.0)) & 63u;
    return (sign << 10u) | (exponent << 6u) | mantissa;
}

vec4 encodeColor(vec3 color) {
    uint r = encodeFloat7_4(color.r);
    uint g = encodeFloat7_4(color.g);
    uint b = encodeFloat6_4(color.b);
    
    uint encoded = (r << 21) | (g << 10) | b;
    return vec4(
        encoded >> 24,
        (encoded >> 16) & 255u,
        (encoded >> 8) & 255u,
        encoded & 255u
    ) / 255.0;
}

void main() {
    discardControlGLPos(gl_FragCoord.xy, glpos);
    gl_FragDepth = gl_FragCoord.z;
    vec4 color = texture(Sampler0, texCoord0) * vertexColor * ColorModulator;
    color.rgb = clamp(color.rgb,0.01,1);
    color.rgb += vec3(glpos.xyz)*0.05;
  //color.rgb = (noise)*100;
    fragColor = color;
//    fragColor = linear_fog(color, vertexDistance,FogStart, FogEnd, FogColor);



//    if(water > 0.9 )    fragColor = linear_fog(color, vertexDistance, -8, FogEnd*0.1, FogColor);

//    if(water > 0.9 )  fragColor.a = 0.75;

    
}
