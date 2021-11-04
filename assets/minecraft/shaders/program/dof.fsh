#version 150

uniform sampler2D DiffuseSampler;
uniform sampler2D PreviousFrameDepthSampler;
uniform sampler2D MainSampler;
uniform sampler2D DepthSampler;

uniform vec2 ScreenSize;
in float near;
in float far;
in float center;
in vec2 texCoord;
in vec2 oneTexel;

out vec4 fragColor;

const vec2 offsets[60] = vec2[60] (vec2(0.0000, 0.2500), vec2(-0.2165, 0.1250), vec2(-0.2165, -0.1250), vec2(-0.0000, -0.2500), vec2(0.2165, -0.1250), vec2(0.2165, 0.1250), vec2(0.0000, 0.5000), vec2(-0.2500, 0.4330), vec2(-0.4330, 0.2500), vec2(-0.5000, 0.0000), vec2(-0.4330, -0.2500), vec2(-0.2500, -0.4330), vec2(-0.0000, -0.5000), vec2(0.2500, -0.4330), vec2(0.4330, -0.2500), vec2(0.5000, -0.0000), vec2(0.4330, 0.2500), vec2(0.2500, 0.4330), vec2(0.0000, 0.7500), vec2(-0.2565, 0.7048), vec2(-0.4821, 0.5745), vec2(-0.6495, 0.3750), vec2(-0.7386, 0.1302), vec2(-0.7386, -0.1302), vec2(-0.6495, -0.3750), vec2(-0.4821, -0.5745), vec2(-0.2565, -0.7048), vec2(-0.0000, -0.7500), vec2(0.2565, -0.7048), vec2(0.4821, -0.5745), vec2(0.6495, -0.3750), vec2(0.7386, -0.1302), vec2(0.7386, 0.1302), vec2(0.6495, 0.3750), vec2(0.4821, 0.5745), vec2(0.2565, 0.7048), vec2(0.0000, 1.0000), vec2(-0.2588, 0.9659), vec2(-0.5000, 0.8660), vec2(-0.7071, 0.7071), vec2(-0.8660, 0.5000), vec2(-0.9659, 0.2588), vec2(-1.0000, 0.0000), vec2(-0.9659, -0.2588), vec2(-0.8660, -0.5000), vec2(-0.7071, -0.7071), vec2(-0.5000, -0.8660), vec2(-0.2588, -0.9659), vec2(-0.0000, -1.0000), vec2(0.2588, -0.9659), vec2(0.5000, -0.8660), vec2(0.7071, -0.7071), vec2(0.8660, -0.5000), vec2(0.9659, -0.2588), vec2(1.0000, -0.0000), vec2(0.9659, 0.2588), vec2(0.8660, 0.5000), vec2(0.7071, 0.7071), vec2(0.5000, 0.8660), vec2(0.2588, 0.9659));//lens properties
#define focal  1.2
#define aperture 1.8
#define MANUAL_FOCUS 64.0
//#define dof
#define dof1




float ld(float depth) {
	return (2.0 * near) / (far + near - depth * (far - near));		// (-depth * (far - near)) = (2.0 * near)/ld - far - near
}

const float PI_3 = 1.0471975512;
#define hexa(k) vec2(cos(PI_3 * k), sin(PI_3 * k))
const vec2 paint_offsets[6] = vec2[6] (hexa(0.), hexa(1.), hexa(2.), hexa(3.), hexa(4.), hexa(5.));
float luma(vec3 color) {
	return dot(color, vec3(0.299, 0.587, 0.114));
}


#define DISPLAY_GAMMA 1.8

#define GOLDEN_ANGLE 2.39996323
#define MAX_BLUR_SIZE 20.0

// Smaller = nicer blur, larger = faster
#define RAD_SCALE 0.7

#define uFar far

float getBlurSize(float depth, float focusPoint, float focusScale)
{
	float coc = clamp((1.0 / focusPoint - 1.0 / depth)*focusScale, -1.0, 1.0);
    return abs(coc) * MAX_BLUR_SIZE;
}

vec3 depthOfField(vec2 texCoord, float focusPoint, float focusScale)
{
    vec4 Input = texture(DiffuseSampler, texCoord).rgba;
    Input.a = ld(texture(PreviousFrameDepthSampler, texCoord).r);
    float centerDepth = Input.a * uFar;
    float centerSize = getBlurSize(centerDepth, focusPoint, focusScale);
    vec3 color = Input.rgb;
    float tot = 1.0;
    
    vec2 texelSize = 1.0 / ScreenSize.xy;

    float radius = RAD_SCALE;
    for (float ang = 0.0; radius < MAX_BLUR_SIZE; ang += GOLDEN_ANGLE)
    {
        vec2 tc = texCoord + vec2(cos(ang), sin(ang)) * texelSize * radius;
        
        vec4 sampleInput = texture(DiffuseSampler, tc).rgba;
        sampleInput.a = ld(texture(PreviousFrameDepthSampler, tc).r);

        vec3 sampleColor = sampleInput.rgb;
        float sampleDepth = sampleInput.a * uFar;
        float sampleSize = getBlurSize(sampleDepth, focusPoint, focusScale);
        
        if (sampleDepth > centerDepth)
        {
        	sampleSize = clamp(sampleSize, 0.0, centerSize*2.0);
        }

        float m = smoothstep(radius-0.5, radius+0.5, sampleSize);
        color += mix(color/tot, sampleColor, m);
        tot += 1.0;
        radius += RAD_SCALE/radius;
    }
    
    return color /= tot;
}


void main() {
	vec3 currentColor = texture(DiffuseSampler, texCoord).rgb;
	float d1 = (texture(DepthSampler, texCoord).r);
	float d2 = (texture(PreviousFrameDepthSampler, texCoord).r);
	gl_FragDepth = mix(d1, d2, 0.95);
	float midDepth = ld(texture(PreviousFrameDepthSampler, vec2(0.5, 0.5)).r) * far;

	//midDepth = MANUAL_FOCUS;

	float aspectRatio = ScreenSize.x / ScreenSize.y;

	float depth = ld(texture(DepthSampler, texCoord).r) * far;
	vec3 color = texture(DiffuseSampler, texCoord).rgb;
	vec3 bcolor = vec3(0.);
	vec3 bcolor2 = vec3(0.);
#ifdef dof
#ifdef dof1

	float nb = 0.0;
	vec2 bcoord = vec2(0.0);
		/*--------------------------------*/

	bcolor = color;
	float pcoc = min(abs(aperture * (focal / 100.0 * (depth - midDepth)) / (depth * (midDepth - focal / 100.0))), oneTexel.x * 15.0);
	pcoc *= float(depth > midDepth);

	vec3 t = vec3(0.0);
	float h = 1.004;
	vec2 d = vec2(pcoc / aspectRatio, pcoc);

	for(int i = 0; i < 60; i++) {

		bcolor += texture(DiffuseSampler, texCoord.xy + offsets[i] * pcoc * vec2(1.0, aspectRatio)).rgb;

	}
	color = bcolor / 61.0;


    //float focusPoint = 78.0;
    //float focusScale = 2.0;
    
    //color.rgb = depthOfField(texCoord, midDepth, focusScale);

#endif

#ifdef dof2	

	bcolor = color;
	bcolor2 = color;

	float pcoc = min(abs(aperture * (focal / 100.0 * (depth - midDepth)) / (depth * (midDepth - focal / 100.0))), oneTexel.x * 15.0);
	pcoc *= float(depth > midDepth);
		//		  pcoc =  pow(pcoc,2)*100;
	vec3 t = vec3(0.0);
	float v = 0.004; // spot size
	pcoc = clamp(pcoc, 0.0, v);

	vec2 d = vec2(v / aspectRatio, v);
	//vec2 d = vec2(pcoc / aspectRatio, pcoc);

	vec2 xy = vec2(texCoord.x, 1.0f - texCoord.y);

	for(int i = 0; i < 6; ++i) {
		bcolor = texture(DiffuseSampler, texCoord.xy + paint_offsets[i] * d).rgb;
		t = max(sign(bcolor - bcolor2), 0.0);

		bcolor2 += (bcolor - bcolor2) * t;
	}
	color = bcolor2;

#endif

#endif

	fragColor = vec4(vec3(color), 1.0);
}
