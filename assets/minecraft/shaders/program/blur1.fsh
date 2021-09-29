#version 150

uniform sampler2D DiffuseSampler;
uniform sampler2D DiffuseDepthSampler;
uniform sampler2D MainSampler;
uniform sampler2D BloomSampler;
uniform vec2 ScreenSize;
out vec4 fragColor;

in vec2 texCoord;





vec2 unpackUnorm2x4(float pack) {
	vec2 xy; xy.x = modf(pack * 255.0 / 16.0, xy.y);
	return xy * vec2(16.0 / 15.0, 1.0 / 15.0);
}

#define SAMPLE_OFFSET 5.

#define INTENSITY 1.

void main() {
float Intensity = 1.0;
float BlurSize = 24.0;
    float mod2 = gl_FragCoord.x + gl_FragCoord.y;
    float res2 = mod(mod2, 2.0f);
    vec2 oneTexel = 1/ScreenSize;

    vec2 uv = gl_FragCoord.xy / ScreenSize.xy *4. -1.5;
    vec2 res = ScreenSize.xy;
	
    float i = SAMPLE_OFFSET;
    i = i * sin(1 * 0.5 + vec3(0, 0, 0)).x; // make this animated
    
     vec3 col = texture( DiffuseSampler, uv).rgb / 2.0;
    col += texture( DiffuseSampler, uv + vec2( i, i ) / res ).rgb / 8.0;
    col += texture( DiffuseSampler, uv + vec2( i, -i ) / res ).rgb / 8.0;
    col += texture( DiffuseSampler, uv + vec2( -i, i ) / res ).rgb / 8.0;
    col += texture( DiffuseSampler, uv + vec2( -i, -i ) / res ).rgb / 8.0;



	fragColor= vec4((col), 1.0);
}
