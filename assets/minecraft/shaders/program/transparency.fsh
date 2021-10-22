#version 150

uniform sampler2D DiffuseSampler;
uniform sampler2D DiffuseDepthSampler;
uniform sampler2D TranslucentSampler;
uniform sampler2D TranslucentDepthSampler;
uniform sampler2D ItemEntitySampler;
uniform sampler2D ItemEntityDepthSampler;
uniform sampler2D ParticlesSampler;
uniform sampler2D ParticlesDepthSampler;
uniform sampler2D WeatherSampler;
uniform sampler2D WeatherDepthSampler;
uniform sampler2D CloudsSampler;
uniform sampler2D CloudsDepthSampler;
uniform sampler2D TranslucentSpecSampler;
uniform vec2 ScreenSize;
in vec2 texCoord;

#define NUM_LAYERS 6

vec4 color_layers[NUM_LAYERS];
float depth_layers[NUM_LAYERS];
int index_layers[NUM_LAYERS] = int[NUM_LAYERS] (0, 1, 2, 3, 4, 5);
int active_layers = 0;

out vec4 fragColor;
vec4 toLinear(vec4 sRGB) {
    return vec4(sRGB.rgb * (sRGB.rgb * (sRGB.rgb * 0.305306011 + 0.682171111) + 0.012522878), sRGB.a);
}
// avoid hardware interpolation
vec4 sample_biquadratic_exact(sampler2D channel, vec2 uv) {
    vec2 res = (textureSize(channel, 0).xy);
    vec2 q = fract(uv * res);
    ivec2 t = ivec2(uv * res);
    const ivec3 e = ivec3(-1, 0, 1);
    vec4 s00 = texelFetch(channel, t + e.xx, 0);
    vec4 s01 = texelFetch(channel, t + e.xy, 0);
    vec4 s02 = texelFetch(channel, t + e.xz, 0);
    vec4 s12 = texelFetch(channel, t + e.yz, 0);
    vec4 s11 = texelFetch(channel, t + e.yy, 0);
    vec4 s10 = texelFetch(channel, t + e.yx, 0);
    vec4 s20 = texelFetch(channel, t + e.zx, 0);
    vec4 s21 = texelFetch(channel, t + e.zy, 0);
    vec4 s22 = texelFetch(channel, t + e.zz, 0);
    vec2 q0 = (q + 1.0) / 2.0;
    vec2 q1 = q / 2.0;
    vec4 x0 = mix(mix(s00, s01, q0.y), mix(s01, s02, q1.y), q.y);
    vec4 x1 = mix(mix(s10, s11, q0.y), mix(s11, s12, q1.y), q.y);
    vec4 x2 = mix(mix(s20, s21, q0.y), mix(s21, s22, q1.y), q.y);
    return mix(mix(x0, x1, q0.x), mix(x1, x2, q1.x), q.x);
}

void try_insert(vec4 color, sampler2D dSampler) {
    if(color.a == 0.0) {
        return;
    }

    float depth = texture(dSampler, texCoord).r;
    color_layers[active_layers] = color;
    depth_layers[active_layers] = depth;

    int jj = active_layers++;
    int ii = jj - 1;
    while(jj > 0 && depth > depth_layers[index_layers[ii]]) {
        int indexTemp = index_layers[ii];
        index_layers[ii] = index_layers[jj];
        index_layers[jj] = indexTemp;

        jj = ii--;
    }
}

vec3 blend(vec3 dst, vec4 src) {
    return (dst * (1.0 - src.a)) + src.rgb;
}

void main() {
    float aspectRatio = ScreenSize.x / ScreenSize.y;
    color_layers[0] = vec4(texture(DiffuseSampler, texCoord).rgb, 1.0);
    depth_layers[0] = texture(DiffuseDepthSampler, texCoord).r;
    active_layers = 1;

//    try_insert( toLinear(texture( CloudsSampler, texCoord )), CloudsDepthSampler);
    try_insert((texture(TranslucentSampler, texCoord)), TranslucentDepthSampler);
    try_insert((texture(ParticlesSampler, texCoord)), ParticlesDepthSampler);
    try_insert(toLinear(texture(WeatherSampler, texCoord)), WeatherDepthSampler);
    try_insert(toLinear(texture(ItemEntitySampler, texCoord)), ItemEntityDepthSampler);

    vec3 texelAccum = color_layers[index_layers[0]].rgb;
    for(int ii = 1; ii < active_layers; ++ii) {
        texelAccum = blend(texelAccum, color_layers[index_layers[ii]]);
    }
//    if(texture( TranslucentSampler, texCoord ).a *255 == 200 &&  texture(DiffuseDepthSampler, texCoord ).r >=1) texelAccum.rgb = vec3(texture( TranslucentSampler, texCoord ).rgb);
    fragColor = vec4(texelAccum.rgb, texture(DiffuseSampler, texCoord).a);
}