#version 150

uniform sampler2D DiffuseSampler;
uniform vec2 ScreenSize;
out vec4 fragColor;

#define SAMPLE_OFFSET 5.
#define INTENSITY 1.

void main() {

    vec2 uv = gl_FragCoord.xy / ScreenSize.xy / 2. + .25;

    vec2 res = ScreenSize.xy;

    float i = SAMPLE_OFFSET;
    
    vec3 col = texture(DiffuseSampler, uv + vec2(i, i) / res).rgb / 6.0;
    col += texture(DiffuseSampler, uv + vec2(i, -i) / res).rgb / 6.0;
    col += texture(DiffuseSampler, uv + vec2(-i, i) / res).rgb / 6.0;
    col += texture(DiffuseSampler, uv + vec2(-i, -i) / res).rgb / 6.0;

    col += texture(DiffuseSampler, uv + vec2(0, i * 2.0) / res).rgb / 12.0;
    col += texture(DiffuseSampler, uv + vec2(i * 2., 0) / res).rgb / 12.0;
    col += texture(DiffuseSampler, uv + vec2(-i * 2., 0) / res).rgb / 12.0;
    col += texture(DiffuseSampler, uv + vec2(0, -i * 2.) / res).rgb / 12.0;

    fragColor = vec4((col), 1.0);
}
