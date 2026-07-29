const PARTICIPANT_REPORT_VOICE_PATTERNS: RegExp[] = [
  /(?:^|[\s，。；：！？])(?:该)?用户(?:表示|认为|希望|担心|喜欢|偏好|需要|要求|提到|说道?|正在|计划|承诺|答应|应该|会|的|与|和)/u,
  /(?:^|[\s，。；：！？])(?:该)?助手(?:表示|认为|希望|担心|需要|要求|提到|说道?|正在|计划|承诺|答应|应该|会|的|与|和)/u,
  /\b(?:the user|the assistant)\b/i,
];

export function hasParticipantReportVoice(text: string): boolean {
  return PARTICIPANT_REPORT_VOICE_PATTERNS.some((pattern) => pattern.test(text));
}
