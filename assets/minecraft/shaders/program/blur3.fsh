#version 150

uniform sampler2D DiffuseSampler;
uniform vec2 ScreenSize;
out vec4 fragColor;
in vec2 texCoord;

#define SAMPLE_OFFSET 5.
#define INTENSITY 1.
float interleaved_gradientNoise() {
    return fract(52.9829189 * fract(0.06711056 * gl_FragCoord.x + 0.00583715 * gl_FragCoord.y));
}
void main() {
    vec2 uv = vec2(gl_FragCoord.xy / (ScreenSize.xy * 2.0));
    vec2 halfpixel = 0.5 / (ScreenSize.xy * 2.0);
    float offset = 25.0*interleaved_gradientNoise();

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
