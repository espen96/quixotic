#version 150

in vec2 texCoord;
in vec2 oneTexel;
in mat4 gbufferProjection;
in mat4 gbufferProjectionInverse;
in mat4 gbufferModelView;
in mat4 gbufferModelViewInverse;
in mat4 gbufferPreviousProjection;
in mat4 gbufferPreviousModelView;
in mat4 projInv;
in mat4 projInv2;
in vec3 currChunkOffset;
in vec3 prevChunkOffset;
in float near;
in float far;
in vec3 prevPosition;

uniform sampler2D DiffuseSampler;
uniform sampler2D CurrentFrameDepthSampler;
uniform vec2 ScreenSize;
uniform sampler2D PreviousFrameSampler;
uniform sampler2D PreviousFrameDepthSampler;

out vec4 fragColor;

#define BLEND_FACTOR 0.65 //[0.01 0.02 0.03 0.04 0.05 0.06 0.08 0.1 0.12 0.14 0.16] higher values = more flickering but sharper image, lower values = less flickering but the image will be blurrier
#define MOTION_REJECTION 0.9 //[0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.5] //Higher values=sharper image in motion at the cost of flickering
#define ANTI_GHOSTING 0.9 //[0.0 0.25 0.5 0.75 1.0] High values reduce ghosting but may create flickering
#define FLICKER_REDUCTION 0.0  //[0.0 0.25 0.5 0.75 1.0] High values reduce flickering but may reduce sharpness


vec3 calculateWorldPos(float depth, vec2 texCoord, mat4 projMat, mat4 modelViewMat) {
    
    vec4 clip = vec4(texCoord * 2 - 1, depth, 1);
    vec4 viewSpace = inverse(projMat) * clip;
    viewSpace /= viewSpace.w;
    return (inverse(modelViewMat) * viewSpace).xyz;
}


	float MinDepth3x3(vec2 uv) {
		float minDepth = texture(CurrentFrameDepthSampler, uv).x;
		for (int x = -1; x <= 1; ++x) {
			for (int y = -1; y <= 1; ++y) {
				if (x == 0 && y == 0) { continue; }
				minDepth = min(minDepth, texture(CurrentFrameDepthSampler, uv + oneTexel * vec2(x, y)).x);
			}
		}
		return minDepth;
	}
vec3 ViewSpaceFromScreenSpace(vec3 pos, mat4 inverseProjection) {
	pos = pos * 2.0 - 1.0;
	#ifdef TAA
		pos.xy -= taaOffset;
	#endif

	vec3 viewPosition  = vec3(vec2(inverseProjection[0].x, inverseProjection[1].y) * pos.xy + inverseProjection[3].xy, inverseProjection[3].z);
	     viewPosition /= inverseProjection[2].w * pos.z + inverseProjection[3].w;

	return viewPosition;
}
vec3 ScreenSpaceFromViewSpace(vec3 viewPosition, mat4 projection) {
	vec3 screenPosition = vec3(projection[0].x, projection[1].y, projection[2].z) * viewPosition + projection[3].xyz;
	#ifdef TAA
		screenPosition.xy -= taaOffset * viewPosition.z;
	#endif
	return screenPosition * (0.5 / -viewPosition.z) + 0.5;
}

float luma(vec3 color) {
	return dot(color,vec3(0.21, 0.72, 0.07));
}
	vec3 Reproject(vec3 position) {
		if (position.z >= 1.0) {
			position = ViewSpaceFromScreenSpace(position, gbufferProjectionInverse);
			position = mat3(gbufferModelViewInverse) * position;
			position = mat3(gbufferPreviousModelView) * position;
			position = ScreenSpaceFromViewSpace(position, gbufferPreviousProjection);
		} else if (position.z > 0.6) {
			position = ViewSpaceFromScreenSpace(position, gbufferProjectionInverse);
			position = mat3(gbufferModelViewInverse) * position + gbufferModelViewInverse[3].xyz;
			position = position + currChunkOffset - prevChunkOffset;
			position = mat3(gbufferPreviousModelView) * position + gbufferPreviousModelView[3].xyz;
			position = ScreenSpaceFromViewSpace(position, gbufferPreviousProjection);
		}

		return position;
	}
// From a presentation given by Lasse Jon Fuglsang Pedersen titled "Temporal Reprojection Anti-Aliasing in INSIDE"
// https://www.youtube.com/watch?v=2XXS5UyNjjU&t=434s
vec3 clipColor(vec3 aabbMin, vec3 aabbMax, vec3 prevColor) {
    // Center of the clip space
    vec3 pClip = (aabbMax + aabbMin) / 2;
    // Size of the clip space
    vec3 eClip = (aabbMax - aabbMin) / 2;

    // The relative coordinates of the previous color in the clip space
    vec3 vClip = prevColor - pClip;
    // Normalized clip space coordintes
    vec3 vUnit = vClip / eClip;
    // The distance of the previous color from the center of the clip space in each axis in the normalized clip space
    vec3 aUnit = abs(vUnit);
    // The divisor is the largest distance from the center along each axis
    float divisor = max(aUnit.x, max(aUnit.y, aUnit.z));
    if (divisor > 1) {
        // If the divisor is larger, than 1, that means that the previous color is outside of the clip space
        // If we divide by divisor, we'll put it into clip space
        return pClip + vClip / divisor;
    }
    // Otherwise it's already clipped
    return prevColor;
}

void main() {
    vec3 currColor = texture(DiffuseSampler, texCoord).rgb;
    fragColor = vec4(currColor, 1);
    float depth = texture(CurrentFrameDepthSampler, texCoord).r ;
if(depth <1.0){
        vec3 position = vec3(texCoord, depth);
		vec3 reprojectedPosition = Reproject(position);

    // We'll recreate the current world position from the texture coord and depth sampler
    float currDepth = texture(CurrentFrameDepthSampler, texCoord).r * 2 - 1;
    vec3 worldPos = calculateWorldPos(currDepth, texCoord, gbufferProjection, gbufferModelView);
    // Then we offset this by the amount the player moved between the two frames
    vec3 prevRelativeWorldPos = worldPos - prevPosition;

    // We can then convert this into the texture coord of the fragment in the previous frame
    vec4 prevClip = gbufferPreviousProjection * gbufferPreviousModelView * vec4(prevRelativeWorldPos, 1);
    vec2 prevTexCoord = (prevClip.xy / prevClip.w + 1) / 2;
 //   reprojectedPosition.xy = prevTexCoord;

    // Throw away the previous data if the uvs fall outside of the screen area
    if (any(greaterThan(abs(reprojectedPosition.xy - 0.5), vec2(0.5)))) {
        return;
    }


    // Temporal antialiasing from same talk mentioned earlier
    vec3 prevColor = texture(PreviousFrameSampler, reprojectedPosition.xy).rgb;
    // We'll calculate the color space from the neighbouring texels
    vec3 minCol = vec3(1);
    vec3 maxCol = vec3(0);
    for (float x = -1; x <= 1; x++) {
        for (float y = -1; y <= 1; y++) {
            vec3 neighbor = texture(DiffuseSampler, texCoord + vec2(x, y) * oneTexel).rgb;
            minCol = min(minCol, neighbor);
            maxCol = max(maxCol, neighbor);
        }
    }
	//to reduce error propagation caused by interpolation during history resampling, we will introduce back some aliasing in motion
	vec2 d = 0.5-abs(fract(reprojectedPosition.xy*vec2(ScreenSize)-texCoord*vec2(ScreenSize))-0.5);
	float mixFactor = dot(d,d);
	float rej = mixFactor*MOTION_REJECTION;

    vec3 clippedPrevColor = clipColor(minCol, maxCol, prevColor);

	vec3 albedoPrev = texture(PreviousFrameSampler, reprojectedPosition.xy).xyz;

	float isclamped = distance(albedoPrev,clippedPrevColor)/luma(albedoPrev);

	//reduces blending factor if current texel is far from history, reduces flickering
	float lumDiff2 = distance(albedoPrev,currColor)/luma(albedoPrev);
	lumDiff2 = 1.0-clamp(lumDiff2*lumDiff2,0.,1.)*FLICKER_REDUCTION;


    // Then we'll clip the previous color into the clip space
    
    vec3 supersampled =  mix(clippedPrevColor,currColor,clamp(BLEND_FACTOR*lumDiff2+rej+isclamped*ANTI_GHOSTING+0.01,0.0,1.0));

    // And use the clipped value for aliasing
    fragColor.rgb = mix(fragColor.rgb, clippedPrevColor, 0.75);
    fragColor.rgb = supersampled;
//    fragColor.a = clamp(BLEND_FACTOR*lumDiff2+rej+isclamped*ANTI_GHOSTING+0.01,0.,1.);

}

}