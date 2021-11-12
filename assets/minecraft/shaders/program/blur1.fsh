#version 150
uniform sampler2D DiffuseSampler;
uniform vec2 ScreenSize;
out vec4 fragColor;

#define SAMPLE_OFFSET 5.
#define INTENSITY 1.

void main() {
    vec2 uv = vec2(gl_FragCoord.xy / (ScreenSize.xy / 2.0));

    vec2 halfpixel = 0.5 / (ScreenSize.xy / 2.0);
    float offset = 15.0;

    vec4 sum = texture(DiffuseSampler, uv) * 4.0;
    sum += texture(DiffuseSampler, uv - halfpixel.xy * offset);
    sum += texture(DiffuseSampler, uv + halfpixel.xy * offset);
    sum += texture(DiffuseSampler, uv + vec2(halfpixel.x, -halfpixel.y) * offset);
    sum += texture(DiffuseSampler, uv - vec2(halfpixel.x, -halfpixel.y) * offset);
    vec4 col = sum / 8.0;
    col = pow(col,vec4(2.2))*2.0;
    col = max(vec4(0.0), col - 0.01);

    fragColor = col;
}
