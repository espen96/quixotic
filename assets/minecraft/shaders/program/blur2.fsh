#version 150

uniform sampler2D DiffuseSampler;
uniform vec2 ScreenSize;
out vec4 fragColor;

#define SAMPLE_OFFSET 5.
#define INTENSITY 1.

void main() {
    vec2 uv = gl_FragCoord.xy / ScreenSize.xy * 2. - .5;
    vec2 res = ScreenSize.xy;

    float i = SAMPLE_OFFSET;
    i = i * sin(1 * 0.5 + vec3(0, 0, 0)).x;

    vec3 col = texture(DiffuseSampler, uv).rgb / 2.0;
    col += texture(DiffuseSampler, uv + vec2(i, i) / res).rgb / 8.0;
    col += texture(DiffuseSampler, uv + vec2(i, -i) / res).rgb / 8.0;
    col += texture(DiffuseSampler, uv + vec2(-i, i) / res).rgb / 8.0;
    col += texture(DiffuseSampler, uv + vec2(-i, -i) / res).rgb / 8.0;

    fragColor = vec4((col), 1.0);
//	fragColor= vec4(vec3(texture(DiffuseSampler, texCoord).a), 1.0);
}
