#version 150

uniform sampler2D DiffuseSampler;
uniform vec2 ScreenSize;
out vec4 fragColor;

#define SAMPLE_OFFSET 5.
#define INTENSITY 1.

void main() {

    vec2 uv = vec2(gl_FragCoord.xy / (ScreenSize.xy * 2.0));
    vec2 halfpixel = 0.5 / (ScreenSize.xy * 2.0);
    float offset = 20.0;

    vec4 sum = texture(DiffuseSampler, uv +vec2(-halfpixel.x * 2.0, 0.0) * offset);
    
    sum += texture(DiffuseSampler, uv + vec2(-halfpixel.x, halfpixel.y) * offset) * 2.0;
    sum += texture(DiffuseSampler, uv + vec2(0.0, halfpixel.y * 2.0) * offset);
    sum += texture(DiffuseSampler, uv + vec2(halfpixel.x, halfpixel.y) * offset) * 2.0;
    sum += texture(DiffuseSampler, uv + vec2(halfpixel.x * 2.0, 0.0) * offset);
    sum += texture(DiffuseSampler, uv + vec2(halfpixel.x, -halfpixel.y) * offset) * 2.0;
    sum += texture(DiffuseSampler, uv + vec2(0.0, -halfpixel.y * 2.0) * offset);
    sum += texture(DiffuseSampler, uv + vec2(-halfpixel.x, -halfpixel.y) * offset) * 2.0;

    fragColor = sum / 12.0;
}
