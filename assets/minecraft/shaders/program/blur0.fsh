#version 150
uniform sampler2D DiffuseSampler;
uniform sampler2D DepthSampler;
uniform sampler2D MainSampler;
uniform vec2 ScreenSize;
out vec4 fragColor;
in vec2 texCoord;
#define SAMPLE_OFFSET 5.
#define INTENSITY 1.
uniform vec2 OutSize;
float map(float value, float min1, float max1, float min2, float max2)
{
    return min2 + (value - min1) * (max2 - min2) / (max1 - min1);
}

vec4 textureGatherOffsets(sampler2D sampler, vec2 texCoord, ivec2[4] offsets, int channel)
{
    ivec2 coord = ivec2(gl_FragCoord.xy);
    return vec4(
        texelFetch(sampler, coord + offsets[0], 0)[channel], texelFetch(sampler, coord + offsets[1], 0)[channel],
        texelFetch(sampler, coord + offsets[2], 0)[channel], texelFetch(sampler, coord + offsets[3], 0)[channel]);
}
    #define sssMin 22
    #define sssMax 47
    #define lightMin 48
    #define lightMax 72
    #define roughMin 73
    #define roughMax 157
    #define metalMin 158
    #define metalMax 251
vec4 pbr(vec2 in1, vec2 in2, vec3 test)
{
    float mod2 = gl_FragCoord.x + gl_FragCoord.y;
    float res = mod(mod2, 2.0f);

    vec4 alphatest = vec4(0.0);
    vec4 pbr = vec4(0.0);

    float maps1 = mix(in1.x, in2.x, res);
    float maps2 = mix(in2.x, in1.x, res);

    maps1 = map(maps1, 0, 1, 128, 255);
    if (maps1 == 128)
        maps1 = 0.0;
    maps2 = map(maps2, 0, 1, 0, 128);

    float maps = in1.x;
    float expanded = int(maps * 255);

    if (expanded >= lightMin && expanded <= lightMax)
        alphatest.r = maps; // Emissives
    float emiss = map(alphatest.r * 255, lightMin, lightMax, 0, 1);

    pbr = vec4(emiss);

    return pbr;
}

void main()
{
    vec2 uv = vec2(gl_FragCoord.xy / (ScreenSize.xy / 2.0));

    vec2 halfpixel = 0.5 / (ScreenSize.xy / 2.0);
    float offset = 25.0;
    float mod2 = gl_FragCoord.x + gl_FragCoord.y;
    float res = mod(mod2, 2.0f);
    vec2 oneTexel = 1 / ScreenSize;
    ivec2 texoffsets[4] = ivec2[](ivec2(0, 1), ivec2(1, 0), -ivec2(0, 1), -ivec2(1, 0));
    vec4 lmgather = textureGatherOffsets(DiffuseSampler, texCoord, texoffsets, 3);

    vec4 OutTexel3 = (texture(MainSampler, texCoord).rgba);
    float depth = (texture(DepthSampler, texCoord).x);
    vec4 cbgather = textureGatherOffsets(MainSampler, texCoord, texoffsets, 2);
    vec4 crgather = textureGatherOffsets(MainSampler, texCoord, texoffsets, 0);
    vec4 pbr = clamp(pbr(OutTexel3.aa, (lmgather.xx), OutTexel3.rgb),0,1);

    float lmx = clamp(mix(OutTexel3.b, dot(cbgather, vec4(1.0)) / 4, res), 0.0, 1);
    float lmy = (clamp(mix(OutTexel3.r, dot(crgather, vec4(1.0)) / 4, res)*2-1, 0.01, 1) );
    if(depth >= 1.0) lmy = 0.6;
    vec4 sum = texture(DiffuseSampler, texCoord) * (lmy+pbr.x)*2;

    fragColor = vec4(sum);
}
