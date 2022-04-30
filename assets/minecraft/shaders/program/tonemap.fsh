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

void getNightDesaturation(inout vec3 color)
{
    float lum = dot(color, vec3(0.15, 0.3, 0.55));
    float lum2 = dot(color, vec3(0.85, 0.7, 0.45)) * 0.5;
    float rodLum = lum2 * 300.0;
    float rodCurve = mix(1.0, rodLum / (2.5 + rodLum), (Purkinje_strength));
    color = mix(lum * 0.5 * vec3(Purkinje_R, Purkinje_G, Purkinje_B), color, rodCurve);

    float brightness = dot(color, vec3(0.2627, 0.6780, 0.0593));
    float amount = clamp(0.15 / (pow(brightness * ndeSat, 2.0) + 0.02), 0, 1);
    vec3 desatColor = mix(color, vec3(brightness), vec3(0.9)) * vec3(0.2, 1.0, 2.0);

    color = mix(color, desatColor, amount);
}

void BSLTonemap(inout vec3 color)
{
    color = EXPOSURE * color;
    color = color / pow(pow(color, vec3(TONEMAP_WHITE_CURVE)) + 1.0, vec3(1.0 / TONEMAP_WHITE_CURVE));
    color = pow(color, mix(vec3(TONEMAP_LOWER_CURVE), vec3(TONEMAP_UPPER_CURVE), sqrt(color)));
}

void main()
{

    vec3 color = texture(DiffuseSampler, texCoord).rgb;

    float vignette = (1.5 - dot(texCoord - 0.5, texCoord - 0.5) * 2.);
    vec2 uv = vec2(gl_FragCoord.xy / (ScreenSize.xy * 2.0));

    getNightDesaturation(color.rgb);

    getNightDesaturation(color.rgb, clamp((lmx + lmy), 0.0, 5));	
    //color = fin * lightScat;
    BSLTonemap(color);

    float lumC = luma(color);
    vec3 diff = color - lumC;
    color = color + diff * (-lumC * CROSSTALK + SATURATION);

    fragColor = vec4((vec3(color.rgb)), 1.0);
}
