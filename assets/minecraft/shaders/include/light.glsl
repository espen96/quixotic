#version 150

#define MINECRAFT_LIGHT_POWER   (0.6)
#define MINECRAFT_AMBIENT_LIGHT (0.4)

#define Ambient_Mult 1.0 
#define Sky_Brightness 1.0 
#define MIN_LIGHT_AMOUNT 1.0 
#define TORCH_AMOUNT 1.0 
#define TORCH_R 1.00 
#define TORCH_G 0.7 
#define TORCH_B 0.5 

vec3 reinhard(vec3 x) {
    x *= 1.66;
    return pow(x / (1.0 + x), vec3(1.0 / 2.2));
}

vec3 ToneMap_Hejl2015(in vec3 hdr) {
    vec4 vh = vec4(hdr * 0.85, 3.0);	//0
    vec4 va = (1.75 * vh) + 0.05;	//0.05
    vec4 vf = ((vh * va + 0.004f) / ((vh * (va + 0.55f) + 0.0491f))) - 0.0821f + 0.000633604888;	//((0+0.004)/((0*(0.05+0.55)+0.0491)))-0.0821
    return vf.xyz / vf.www;
}

float map(float value, float min1, float max1, float min2, float max2) {
    return min2 + (value - min1) * (max2 - min2) / (max1 - min1);
}
vec4 minecraft_mix_light(vec3 lightDir0, vec3 lightDir1, vec3 normal, vec4 color) {
    lightDir0 = normalize(lightDir0);
    lightDir1 = normalize(lightDir1);
    float light0 = max(0.0, dot(lightDir0, normal));
    float light1 = max(0.0, dot(lightDir1, normal));
    float lightAccum = min(1.0, (light0 + light1) * MINECRAFT_LIGHT_POWER + MINECRAFT_AMBIENT_LIGHT);
    return vec4(color.rgb * lightAccum, color.a);
}

vec4 minecraft_sample_lightmap(sampler2D lightMap, ivec2 uv) {
    return texture(lightMap, clamp(uv / 256.0, vec2(0.5 / 16.0), vec2(15.5 / 16.0))); // x is torch, y is sun
}

float luma3(vec3 color) {
    return dot(color, vec3(0.21, 0.72, 0.07));
}

vec4 minecraft_sample_lightmap2(sampler2D lightMap, ivec2 uv) {
    vec3 blocklightColSqrt = vec3(TORCH_R, TORCH_G, TORCH_B);
    vec3 blocklightCol = blocklightColSqrt * blocklightColSqrt;

    vec3 blockLight = vec3(uv.x / 16.0) / 16;
    vec3 skyLight = vec3(uv.y / 16.0) / 16;

    vec3 blockLighting = blocklightCol * blockLight;

    vec2 block = vec2(clamp(vec2(uv.x, 0) / 256.0, vec2(0.5 / 16.0), vec2(15.5 / 16.0)));
    vec2 sky = vec2(clamp(vec2(0, uv.y) / 256.0, vec2(0.5 / 16.0), vec2(15.5 / 16.0)));

    vec4 lm = texture(lightMap, clamp(uv / 256.0, vec2(0.5 / 16.0), vec2(15.5 / 16.0)));
    vec4 sl = texture(lightMap, sky);
    sl *= sl;

    vec4 bl = texture(lightMap, block);

//    bl.rgb *= blockLight;
    bl.rgb *= blocklightCol;
    bl.rgb = mix(bl.rgb, lm.rgb, 0.75);

    vec4 ambient = vec4(bl + sl);
    ambient.rgb = mix(ambient.rgb * ambient.rgb, ambient.rgb, 0.5);

    return vec4(clamp(ambient.rgb, 0.0, 1.0), 1.0);
}