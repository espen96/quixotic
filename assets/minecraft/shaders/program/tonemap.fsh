#version 150

uniform sampler2D DiffuseSampler;
uniform sampler2D MainSampler;
uniform sampler2D BloomSampler;
uniform sampler2D blursampler;
uniform vec2 ScreenSize;
out vec4 fragColor;
in vec4 exposure;
in vec2 rodExposureDepth;
in vec2 texCoord;

    #define EXPOSURE 1.5
    #define TONEMAP_WHITE_CURVE 1.7 
    #define TONEMAP_LOWER_CURVE 1.2 
    #define TONEMAP_UPPER_CURVE 1.3 
    #define CROSSTALK 0.25 // Desaturates bright colors and preserves saturation in darker areas (inverted if negative). Helps avoiding almsost fluorescent colors 
    #define SATURATION 0.25 // Negative values desaturates colors, Positive values saturates color, 0 is no change

    #define ndeSat 7.0
    #define Purkinje_strength 1.0	// Simulates how the eye is unable to see colors at low light intensities. 0 = No purkinje effect at low exposures 
    #define Purkinje_R 0.4
    #define Purkinje_G 0.7 
    #define Purkinje_B 1.0
    #define Purkinje_Multiplier 0.1 // How much the purkinje effect increases brightness

    #define SAMPLE_OFFSET 5.
    #define INTENSITY 0.1

float luma(vec3 color) {
    return dot(color, vec3(0.299, 0.587, 0.114));
}

void getNightDesaturation(inout vec3 color, float lmx) {
    float lum = dot(color, vec3(0.15, 0.3, 0.55));
    float lum2 = dot(color, vec3(0.85, 0.7, 0.45)) *0.5;
    float rodLum = lum2 * 300.0;
    float rodCurve = mix(1.0, rodLum / (2.5 + rodLum), (Purkinje_strength));
    color = mix(lum * 0.5 * vec3(Purkinje_R, Purkinje_G, Purkinje_B), color, rodCurve);

    float brightness = dot(color, vec3(0.2627, 0.6780, 0.0593));
    float amount = clamp(0.15 / (pow(brightness * ndeSat, 2.0) + 0.02), 0, 1);
    vec3 desatColor = mix(color, vec3(brightness), vec3(0.9)) * vec3(0.2, 1.0, 2.0);

    color = mix(color, desatColor, amount);

}

float interleaved_gradientNoise() {
    return fract(52.9829189 * fract(0.06711056 * gl_FragCoord.x + 0.00583715 * gl_FragCoord.y));
}
vec3 int8Dither(vec3 color) {
    float dither = interleaved_gradientNoise();
    return color + dither * exp2(-8.0);
}

void BSLTonemap(inout vec3 color) {
    color = EXPOSURE * color;
    color = color / pow(pow(color, vec3(TONEMAP_WHITE_CURVE)) + 1.0, vec3(1.0 / TONEMAP_WHITE_CURVE));
    color = pow(color, mix(vec3(TONEMAP_LOWER_CURVE), vec3(TONEMAP_UPPER_CURVE), sqrt(color)));
}
vec3 ToneMap_Hejl2015(in vec3 hdr)
{
    vec4 vh = vec4(hdr*0.85, 3.0);	//0
    vec4 va = (1.75 * vh) + 0.05;	//0.05
    vec4 vf = ((vh * va + 0.004f) / ((vh * (va + 0.55f) + 0.0491f))) - 0.0821f+0.000633604888;	//((0+0.004)/((0*(0.05+0.55)+0.0491)))-0.0821
    return vf.xyz / vf.www;
}
vec2 unpackUnorm2x4(float pack) {
    vec2 xy;
    xy.x = modf(pack * 255.0 / 16.0, xy.y);
    return xy * vec2(16.0 / 15.0, 1.0 / 15.0);
}

float cdist(vec2 coord) {
    return max(abs(coord.s - 0.5), abs(coord.t - 0.5)) * 2.0;
}


#define SHARPENING 0.25 //[0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54 0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65 0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76 0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87 0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.0 ]
vec3 LinearTosRGB(in vec3 color)
{
    vec3 x = color * 12.92f;
    vec3 y = 1.055f * pow(clamp(color,0.0,1.0), vec3(1.0f / 2.4f)) - 0.055f;

    vec3 clr = color;
    clr.r = color.r < 0.0031308f ? x.r : y.r;
    clr.g = color.g < 0.0031308f ? x.g : y.g;
    clr.b = color.b < 0.0031308f ? x.b : y.b;

    return clr;
}
vec3 toLinear(vec3 sRGB){
	return sRGB * (sRGB * (sRGB * 0.305306011 + 0.682171111) + 0.012522878);
}
vec4 textureGatherOffsets(sampler2D sampler, vec2 texCoord, ivec2[4] offsets, int channel)
{
    ivec2 coord = ivec2(gl_FragCoord.xy);
    return vec4(
        texelFetch(sampler, coord + offsets[0], 0)[channel], texelFetch(sampler, coord + offsets[1], 0)[channel],
        texelFetch(sampler, coord + offsets[2], 0)[channel], texelFetch(sampler, coord + offsets[3], 0)[channel]);
}
void main() {

    float mod2 = gl_FragCoord.x + gl_FragCoord.y;
    float res = mod(mod2, 2.0f);
    vec2 oneTexel = 1 / ScreenSize;
        ivec2 texoffsets[4] = ivec2[](ivec2(0, 1), ivec2(1, 0), -ivec2(0, 1), -ivec2(1, 0));

        vec4 OutTexel3 = (texture(MainSampler, texCoord).rgba);
        vec4 cbgather = textureGatherOffsets(MainSampler, texCoord, texoffsets, 2);
        vec4 crgather = textureGatherOffsets(MainSampler, texCoord, texoffsets, 0);
        float lmx = clamp(mix(OutTexel3.b, dot(cbgather, vec4(1.0)) / 4, res), 0.0, 1);
        float lmy = clamp(mix(OutTexel3.r, dot(crgather, vec4(1.0)) / 4, res), 0.0, 1);


    vec3 color = texture(DiffuseSampler, texCoord).rgb;




    
    /*
    //Weights : 1 in the center, 0.5 middle, 0.25 corners
    vec3 albedoCurrent1 = texture2D(DiffuseSampler, texCoord + vec2(oneTexel.x,oneTexel.y)/1*0.5).rgb;
    vec3 albedoCurrent2 = texture2D(DiffuseSampler, texCoord + vec2(oneTexel.x,-oneTexel.y)/1*0.5).rgb;
    vec3 albedoCurrent3 = texture2D(DiffuseSampler, texCoord + vec2(-oneTexel.x,-oneTexel.y)/1*0.5).rgb;
    vec3 albedoCurrent4 = texture2D(DiffuseSampler, texCoord + vec2(-oneTexel.x,oneTexel.y)/1*0.5).rgb;


    vec3 m1 = -0.5/3.5*color + albedoCurrent1/3.5 + albedoCurrent2/3.5 + albedoCurrent3/3.5 + albedoCurrent4/3.5;
    vec3 std = abs(color - m1) + abs(albedoCurrent1 - m1) + abs(albedoCurrent2 - m1) +
     abs(albedoCurrent3 - m1) + abs(albedoCurrent3 - m1) + abs(albedoCurrent4 - m1);
    float contrast = 1.0 - luma(std)/5.0;
    color = color*(1.0+(SHARPENING)*contrast)
          - (SHARPENING)/(1.0-0.5/3.5)*contrast*(m1 - 0.5/3.5*color);
    */
    float vignette = (1.5 - dot(texCoord - 0.5, texCoord - 0.5) * 2.);
    vec2 uv = vec2(gl_FragCoord.xy / (ScreenSize.xy * 2.0));
    vec2 halfpixel = 0.5 / (ScreenSize.xy * 2.0);
    float offset = 50.0*interleaved_gradientNoise();

    vec4 sum = texture(blursampler, texCoord) ;
    vec4 lmgather = textureGatherOffsets(DiffuseSampler, texCoord, texoffsets, 3);

    vec3 col = (sum.rgb);


    vec3 fin = col.rgb;

	float lightScat = clamp(5.0*0.05*pow(exposure.a,0.2),0.0,1.0)*vignette;


	color = (color+fin*lightScat);

    getNightDesaturation(color.rgb, clamp((lmx + lmy), 0.0, 5));	
    //color = fin * lightScat;
    BSLTonemap(color);
    //color = ToneMap_Hejl2015(color);
        //color = LinearTosRGB(color);

    float lumC = luma(color);
    vec3 diff = color - lumC;
    color = color + diff * (-lumC * CROSSTALK + SATURATION);

    //color.rgb = vec3(VL_abs);
 

    fragColor = vec4((vec3(color.rgb)), 1.0);

}
